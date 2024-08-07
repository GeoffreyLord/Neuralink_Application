import csv
import math
import logging
import numpy as np
import itertools as it

def exportWeights(model, drgnfly):
    """ All Weights and Bias Put in Single List """
    model_params = []

    for l in range(len(model.layers)):
        for n in range(len(model.layers[l].neurons)):
            for w in range(len(model.layers[l].neurons[n].w)):
                    model_params.append(model.layers[l].neurons[n].w[w].data)

    for l in range(len(model.layers)):
        for n in range(len(model.layers[l].neurons)):
                model_params.append(model.layers[l].neurons[n].b.data)


    """" Write to File """
    #These should be saved to the directory where
    filename = 'Input-' + str(drgnfly.numInputNodes) + '-NodePHidden-' + str(drgnfly.numNodesPerHiddenLayer) + '-NHidden-' + str(drgnfly.numHiddenLayers) + '-NOut-' + str(drgnfly.numOutputNodes)+'.csv'
    logging.warning('Writing Model Weights and Bias to: ' + str(filename))
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in model_params:
            writer.writerow([item])


class modelData:
   def __init__(self, numInputNodes, min, max, loadData, filename):
        self.X = np.zeros((numInputNodes, 0))
        self.Y = np.zeros((1,0))
        if loadData:
            with open(filename, mode ='r')as file:
                csvFile = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)

                #self.X = np.zeros((1,0))
                for lines in csvFile:
                    self.X = np.column_stack([self.X, lines[:-1]])  # Append X data
                    self.Y = np.column_stack([self.Y, lines[-1]])
                self.X = np.transpose(self.X)
                self.Y = self.Y.flatten()
                # for i in range(len(self.Y)):
                #     self.Y[i] = float(self.Y[i])
        else:
            trainingDataNumRows = int((max - min + 1)**numInputNodes)
            self.X = np.zeros((trainingDataNumRows, numInputNodes))
            self.Y = np.zeros((trainingDataNumRows))

   def generateTrainData(self, min, max, function):
        numRows, numCols = self.X.shape
        set_row = np.arange(min, max+1)
        repeats = (numCols, 1)
        set_matrix = np.tile(set_row, repeats)
        cartesian_product = list(it.product(*set_matrix))

        for i in range(numRows):
            for j in range(numCols):
                self.X.itemset((i,j), cartesian_product[i][j])


        char_string_arr = ['var_a','var_b','var_c','var_d','var_e','var_f','var_g','var_h','var_i','var_j','var_k','var_l','var_m','var_n','var_o','var_p','var_q','var_r','var_s','var_t','var_u','var_v','var_w','var_x','var_y','var_z']
        
        for i in range(numCols):
            dataObj = 'self.X.item(i,' + str(i) + ')'
            function = function.replace(char_string_arr[i], dataObj) 

        for i in range(numRows):
            try:
                self.Y[i] = eval(function).real #TODO: This may end up being a really stupid addition I forget about. Changing for testing. Can not yet support complex numbers
            except (ValueError, SyntaxError):
                self.Y[i] = float('nan')

