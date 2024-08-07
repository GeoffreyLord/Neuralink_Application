import os
import time
import torch
import logging
import numpy as np
from management.viz import plotLoss

def testModel(drgnfly, data, model):
    """ Create Testing Data """
    testing_indices = np.random.choice(data.X.shape[0], drgnfly.testingDataSize, replace=False)

    X_test = data.X[testing_indices]
    Y_test = data.Y[testing_indices]

    """ Test Model """
    logging.warning('====================Testing Model====================')
    accuracyCounter = 0


    """ Time of Inference """
    total_time = 0.0

    for k in range(drgnfly.testingDataSize):
        start_time = time.time()
        yPredTest = model(X_test[k])
        end_time = time.time()
        total_time += end_time-start_time
        if (np.abs(yPredTest.data.item() - Y_test[k]) < drgnfly.accuracyThreshold):
            accuracyCounter += 1
        logging.warning('Test - Inputs: {}\t Predicted: {}\t True: {}\t Delta: {}'.format(X_test[k], yPredTest.data.item(), Y_test[k], (np.abs(yPredTest.data.item() - Y_test[k]))))


    time_per_input = (total_time/drgnfly.testingDataSize)*1000
    logging.warning('Inference Time Per Input: {} [ms]'.format(time_per_input))
    modelAccuracy = np.round((accuracyCounter/drgnfly.testingDataSize)*100, 4)
    logging.warning('Model Accuracy: {}%'.format(modelAccuracy))



def trainModel(drgnfly, data, model):
    logging.warning('====================Training Model====================')
    training_indices = np.random.choice(data.X.shape[0], drgnfly.trainingDataSize, replace=False)

    """Optimization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=drgnfly.learningRate)
    scaler = torch.cuda.amp.GradScaler()


    X = data.X[training_indices]
    Y = data.Y[training_indices]

    X = torch.tensor(X, dtype=torch.double).to(device)
    Y = torch.tensor(Y, dtype=torch.double).to(device)
    

    """ Initalize Loss Accumulator """
    lossAccumulator = []

    """ -----Model Training----- """
    start_time = time.time()
    for k in range(drgnfly.numEpochs):

        """ forward pass """
        ypred = [model(x) for x in X]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(Y, ypred))

        """ backward pass """
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lossAccumulator.append(loss.data.item())


        logging.warning('Epoch - {}\tLoss - {}\tInput[0] - {}\tPredicted[0] - {}\tTrue[0] - {}'.format(k, np.round(loss.data.item(), 8), X[0].data.tolist(), ypred[0].data.item(), Y[0]))

    end_time = time.time()
    logging.warning('Model Training Time: {} [m]'.format(np.round((end_time-start_time)/60, 3)))
    
    try:
        os_cmd = 'spd-say "Training Complete"'
        if os.system(os_cmd) != 0:
            raise Exception()
    except:
        os.system('say "Training Complete"')

    #plotLoss(lossAccumulator)
