import numpy as np 
import statistics
import logging
from testing.profileModel import profileModel
import os

""" Relu Function """
def relu(A):
    return ((A) + (np.abs(A)))/2

""" Normalizes the activation trigger to 1 or 0"""
def boundActivation(A):
    if A != 0.0:
        A = 1
    return A

def calcSimplifiedParams(data ,params, drgnfly):
    """ 
    This function can currently suport any N layer model that takes in N input values and outputs one value.
    It can accomidate N number of hidden layer nodes. 
    """
    counter = 0

    """ First Hidden Layer Weights"""
    FH_LayerW = []
    Hidden_Layer_W = []
    for i in range(drgnfly.numNodesPerHiddenLayer):
        temp_param_arr = []
        for j in range(drgnfly.numInputNodes):
            temp_param_arr.append(params[counter])
            counter += 1
        FH_LayerW.append(temp_param_arr)
    Hidden_Layer_W.append(FH_LayerW)

    """ Subsiquent Hidden Layer Weights """
    for p in range(drgnfly.numHiddenLayers-1):
        SH_LayerW = []
        for i in range(drgnfly.numNodesPerHiddenLayer):
            temp_param_arr = []
            for j in range(drgnfly.numNodesPerHiddenLayer):
                temp_param_arr.append(params[counter])
                counter += 1
            SH_LayerW.append(temp_param_arr)
        Hidden_Layer_W.append(SH_LayerW)
    
    """ Output Layer Weights """
    Y_W = []
    for i in range(drgnfly.numNodesPerHiddenLayer):
        Y_W.append(params[counter])
        counter += 1
    
    """ Bias Values """
    Bias = params[counter:]
    Hidden_Bias = np.zeros((drgnfly.numHiddenLayers, drgnfly.numNodesPerHiddenLayer))
    for i in range(drgnfly.numHiddenLayers):
        for j in range(drgnfly.numNodesPerHiddenLayer):
            Hidden_Bias[i,j] = params[counter]
            counter += 1

    """ First Hidden Layer Activations """
    FH_Layer = [] 
    """ Second Hidden Layer Activations """
    Hidden_Layer_Act = []

    """Compute Activation of First Layer """
    for i in range(drgnfly.numNodesPerHiddenLayer):
        temp = 0
        for j in range(drgnfly.numInputNodes):
            temp += data[j]*FH_LayerW[i][j]
        FH_Layer.append(relu(temp  + Bias[i]))
    Hidden_Layer_Act.append(FH_Layer)

    

    """ Compute Activation of Subsiquent Layers """
    for p in range(drgnfly.numHiddenLayers-1):
        SH_Layer = []
        for i in range(drgnfly.numNodesPerHiddenLayer):
            temp = 0
            if p == 0:
                for j in range(drgnfly.numNodesPerHiddenLayer):
                    temp += Hidden_Layer_Act[0][j]*Hidden_Layer_W[1][i][j]
                temp_bias = Bias[i+drgnfly.numNodesPerHiddenLayer]
            else:
                for j in range(drgnfly.numNodesPerHiddenLayer):
                    temp += Hidden_Layer_Act[p][j]*Hidden_Layer_W[p+1][i][j]
                temp_bias = Hidden_Bias[p][i]
            SH_Layer.append(relu(temp + temp_bias))
        Hidden_Layer_Act.append(SH_Layer)

    """ Set all Activations to 0 or 1"""
    for i in range(len(Hidden_Layer_Act)):
        for j in range(len(Hidden_Layer_Act[i])):
            Hidden_Layer_Act[i][j] = boundActivation(Hidden_Layer_Act[i][j])


    """ Calculate the weights and bias of the activated nodes """
    simplified_bias = 0
    
    """ Set Weights and Bias to Zero if not Activated """
    for i in range(len(Hidden_Layer_W)):
        for j in range(len(Hidden_Layer_W[i])):
            for k in range(len(Hidden_Layer_W[i][j])):
                if Hidden_Layer_Act[i][j] == 0:
                    Hidden_Layer_W[i][j][k] = 0
                    Hidden_Bias[i][j] = 0


    """ Allocate weights var for factored weights """
    weights = np.zeros((drgnfly.numInputNodes))

    """ Calculate Factored Weights and Bias of NN """
    if drgnfly.numHiddenLayers == 2:
        for p in range(drgnfly.numInputNodes):
            weight_accum = 0
            simplified_bias = 0
            for i in range(drgnfly.numNodesPerHiddenLayer):
                for j in range(drgnfly.numNodesPerHiddenLayer):
                    weight_accum += Hidden_Layer_W[0][j][p] * Hidden_Layer_W[1][i][j] * Y_W[i]
                    simplified_bias += Hidden_Bias[0][i]*Hidden_Layer_W[1][j][i]*Y_W[j]
                simplified_bias += Hidden_Bias[1][i]*Y_W[i]
            simplified_bias += Bias[-1]
            weights[p] = weight_accum

    elif drgnfly.numHiddenLayers == 3:
        for p in range(drgnfly.numInputNodes):
            weight_accum = 0
            simplified_bias = 0
            for i in range(drgnfly.numNodesPerHiddenLayer):
                for j in range(drgnfly.numNodesPerHiddenLayer):
                    for k in range(drgnfly.numNodesPerHiddenLayer):
                        weight_accum += Hidden_Layer_W[0][k][p] * Hidden_Layer_W[1][j][k] * Hidden_Layer_W[2][i][j] * Y_W[i] 
                        simplified_bias += Hidden_Bias[0][i]*Hidden_Layer_W[1][j][i]*Hidden_Layer_W[2][k][j]*Y_W[k]
                    simplified_bias += Hidden_Bias[1][i]*Hidden_Layer_W[2][j][i]*Y_W[j] 
                simplified_bias += Hidden_Bias[2][i]*Y_W[i] 
            simplified_bias += Bias[-1] 
            weights[p] = weight_accum

    elif drgnfly.numHiddenLayers == 4:
        for p in range(drgnfly.numInputNodes):
            weight_accum = 0
            simplified_bias = 0
            for i in range(drgnfly.numNodesPerHiddenLayer):
                for j in range(drgnfly.numNodesPerHiddenLayer):
                    for k in range(drgnfly.numNodesPerHiddenLayer):
                        for l in range(drgnfly.numNodesPerHiddenLayer):
                            weight_accum += Hidden_Layer_W[0][l][p] * Hidden_Layer_W[1][k][l] * Hidden_Layer_W[2][j][k] * Hidden_Layer_W[3][i][j] * Y_W[i] 
                            simplified_bias += Hidden_Bias[0][i]*Hidden_Layer_W[1][j][i]*Hidden_Layer_W[2][k][j]*Hidden_Layer_W[3][l][k]*Y_W[l]
                        simplified_bias += Hidden_Bias[1][i]*Hidden_Layer_W[2][j][i]*Hidden_Layer_W[3][k][j]*Y_W[k]
                    simplified_bias += Hidden_Bias[2][i]*Hidden_Layer_W[3][j][i]*Y_W[j] 
                simplified_bias += Hidden_Bias[3][i]*Y_W[i] 
            simplified_bias += Bias[-1] 
            weights[p] = weight_accum

    elif drgnfly.numHiddenLayers == 5:
        for p in range(drgnfly.numInputNodes):
            weight_accum = 0
            simplified_bias = 0
            for i in range(drgnfly.numNodesPerHiddenLayer):
                for j in range(drgnfly.numNodesPerHiddenLayer):
                    for k in range(drgnfly.numNodesPerHiddenLayer):
                        for l in range(drgnfly.numNodesPerHiddenLayer):
                            for m in range(drgnfly.numNodesPerHiddenLayer):
                                weight_accum += Hidden_Layer_W[0][m][p] * Hidden_Layer_W[1][l][m] * Hidden_Layer_W[2][k][l] * Hidden_Layer_W[3][j][k] * Hidden_Layer_W[4][i][j] * Y_W[i] 
                                simplified_bias += Hidden_Bias[0][i]*Hidden_Layer_W[1][j][i]*Hidden_Layer_W[2][k][j]*Hidden_Layer_W[3][l][k]* Hidden_Layer_W[4][m][l]*Y_W[m]
                            simplified_bias += Hidden_Bias[1][i]*Hidden_Layer_W[2][j][i]*Hidden_Layer_W[3][k][j]*Hidden_Layer_W[4][l][k]*Y_W[l]
                        simplified_bias += Hidden_Bias[2][i]*Hidden_Layer_W[3][j][i]*Hidden_Layer_W[4][k][j]*Y_W[k]
                    simplified_bias += Hidden_Bias[3][i]*Hidden_Layer_W[4][j][i]*Y_W[j] 
                simplified_bias += Hidden_Bias[4][i]*Y_W[i] 
            simplified_bias += Bias[-1] 
            weights[p] = weight_accum

    return weights, simplified_bias
    

def exportAnalysis(drgnfly, data, bias_arr, weight_arr):
    filename = 'ModelAnalysisOutput.txt'
    if os.path.exists(filename): 
        os.remove(filename)
    logging.warning('Writing Analysis Output to: ' + str(filename))
    with open(filename, 'w', newline='') as file:
        for i in range(len(data.X)):
            predicted = 0
            file.write('-----------------------------\n')
            file.write('Inputs = ' + str(data.X[i]) + '\n')
            file.write('True: ' + str(data.Y[i]) + '\n')
            for h in range(len(data.X[i])):
                predicted += data.X[i][h]*weight_arr[i,h]
            predicted += bias_arr[i]
            file.write('Predicted: '+  str(predicted) + '\n')
            file.write('weights: ' + str(weight_arr[i]) + '\n')
            file.write('simplified_bias: ' + str(bias_arr[i]) + '\n')

        file.write('==========Model Metrics=========\n')
        for h in range(len(data.X[i])):
            file.write('\tNN Weights for input [' + str(h) + ']\n')
            file.write('\t\t mean: '+ str(statistics.mean(weight_arr[:,h])) + '\n')
            file.write('\t\t stdev: '+ str(statistics.stdev(weight_arr[:,h])) + '\n')
        file.write('\tNN Bias\n')
        file.write('\t\t mean: '+ str(statistics.mean(bias_arr)) + '\n')
        file.write('\t\t stdev: '+ str(statistics.stdev(bias_arr)) + '\n')
          


def evaluateModel(drgnfly, data, model):
    """ Variables for Factored Model Parameters"""
    bias_arr = []
    weight_arr = np.zeros((len(data.X), len(data.X[0])))  

    """ Parse Out Model Paramaters"""
    params = []
    for l in range(len(model.layers)):
        for n in range(len(model.layers[l].neurons)):
            for w in range(len(model.layers[l].neurons[n].w)):
                    params.append(model.layers[l].neurons[n].w[w].data.item())

    for l in range(len(model.layers)):
        for n in range(len(model.layers[l].neurons)):
                params.append(model.layers[l].neurons[n].b.data.item())


    logging.warning('==========Test Cases==========')
    for i in range(len(data.X)):
        truth = data.Y[i]
        predicted = 0
        weight, bias = calcSimplifiedParams(data.X[i], params, drgnfly)
        for j in range(len(data.X[0])):
            weight_arr[i,j] = weight[j]
        bias_arr.append(bias)

        """ Prints some test cases to the console if all values are 1 or 0 and their sum is 1 or 0"""
        if (np.sum(data.X[i]) == 0.0) or (np.sum(data.X[i]) == 1.0):
            print_bool = 0
            for k in range(len(data.X[i])):
                if data.X[i,k] == 1 or data.X[i,k] == 0:
                    print_bool += 1
            if print_bool == 3:
                logging.warning('-----------------------------')
                logging.warning('Inputs = ' + str(data.X[i]))
                logging.warning("True: " + str(truth))
                for h in range(len(data.X[i])):
                    predicted += data.X[i][h]*weight[h]
                predicted += bias
                logging.warning('Predicted: '+  str(predicted))
                """ For all input it seems as if negating c makes us more accurate """
                logging.warning('weights: ' + str(weight))
                logging.warning('simplified_bias: ' + str(bias))


        """ For Testing - Can Delete Later"""
        if (len(data.X[0]) == 3):
            if (data.X[i][0] == 2) and (data.X[i][1] == 1) and (data.X[i][2] == 1): 
                logging.warning('-----------------------------')
                logging.warning('Inputs = ' + str(data.X[i]))
                logging.warning("True: " + str(truth))
                for h in range(len(data.X[i])):
                    predicted += data.X[i][h]*weight[h]
                predicted += bias
                logging.warning('Predicted: '+  str(predicted))
                """ For all input it seems as if negating c makes us more accurate """
                logging.warning('weights: ' + str(weight))
                logging.warning('simplified_bias: ' + str(bias)) 


    """ Average Weights and Bias Accross Training Inputs """    
    logging.warning('==========Model Metrics=========')
    for h in range(len(data.X[i])):
        logging.warning('\tNN Weights for input [' + str(h) + ']')
        logging.warning('\t\t mean: '+ str(statistics.mean(weight_arr[:,h])))
        logging.warning('\t\t stdev: '+ str(statistics.stdev(weight_arr[:,h])))
    logging.warning('\tNN Bias')
    logging.warning('\t\t mean: '+ str(statistics.mean(bias_arr)))
    logging.warning('\t\t stdev: '+ str(statistics.stdev(bias_arr)))

    exportAnalysis(drgnfly, data, bias_arr, weight_arr) #TODO: This needs to be updated to write predictions for input functions. 

    profileModel(drgnfly, data, weight_arr, bias_arr)






