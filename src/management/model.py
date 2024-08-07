import os
import time
import logging
import numpy as np
#from micrograd.engine import Value
from management.viz import plotLoss
#from micrograd.nn import Neuron, Layer, MLP

def testModel(drgnfly, data, model):
    """ Create Testing Data """
    testing_indices = np.random.choice(data.X.shape[0], drgnfly.testingDataSize, replace=False)

    X_test = data.X[testing_indices]
    Y_test = data.Y[testing_indices]

    """ Test Model """
    logging.warning('====================Testing Model====================')
    accuracyCounter = 0

    for k in range(drgnfly.testingDataSize):
        yPredTest = model(X_test[k])
        if (np.abs(yPredTest.data - Y_test[k]) < drgnfly.accuracyThreshold):
            accuracyCounter += 1
        logging.warning('Test - Inputs: {}\t Predicted: {}\t True: {}\t Delta: {}'.format(X_test[k], yPredTest.data, Y_test[k], (np.abs(yPredTest.data - Y_test[k]))))

    modelAccuracy = np.round((accuracyCounter/drgnfly.testingDataSize)*100, 4)
    logging.warning('Model Accuracy: {}%'.format(modelAccuracy))



def trainModel(drgnfly, data, model):
    logging.warning('====================Training Model====================')
    training_indices = np.random.choice(data.X.shape[0], drgnfly.trainingDataSize, replace=False)

    X = data.X[training_indices]
    Y = data.Y[training_indices]

    """ Initalize Loss Accumulator """
    lossAccumulator = []

    """ -----Model Training----- """
    start_time = time.time()
    for k in range(drgnfly.numEpochs):
        """ forward pass """
        ypred = [model(x) for x in X]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(Y, ypred))


        """ backward pass """
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        

        """ Alter Learning Rate"""
        # if loss.data < 8 and drgnfly.rateChangeBool == True:
        #     drgnfly.learningRate = drgnfly.learningRate * .5
        #     rateChangeBool = False

        #if loss.data < 0.5:
        #   rateChangeBool = True

        """ update parameters """
        for p in model.parameters():
            p.data += -drgnfly.learningRate * p.grad
        
        lossAccumulator.append(loss.data)
        logging.warning('Epoch - {}\tLoss - {}\tInput[0] - {}\tPredicted[0] - {}\tTrue[0] - {}'.format(k, np.round(loss.data, 8), X[0], ypred[0].data, Y[0]))

    end_time = time.time()
    logging.warning('Model Training Time: {} [m]'.format(np.round((end_time-start_time)/60, 3)))
    
    try:
        os_cmd = 'spd-say "Training Complete"'
        if os.system(os_cmd) != 0:
            raise Exception()
    except:
        os.system('say "Training Complete"')

    #plotLoss(lossAccumulator)
