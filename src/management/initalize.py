import csv
import sys
import yaml
import logging
from management.dataMgmt import modelData
from management.drgnflyObj import DrgnflyObj
from micrograd.nn import MLP
#from micrograd.engine import Value
# from micrograd.nn import Neuron, Layer, MLP

def loadConfigFile(filename, drgnfly):
    config = yaml.safe_load(open(filename))
    logging.warning('====================Model Configuration====================')
    logging.warning('Loaded the Following Model Configuration:')
    print(yaml.dump(config))

    drgnfly.accuracyThreshold = config['accuracyThreshold']
    drgnfly.inputFunct = config['inputFunct']
    drgnfly.learningRate = config['learningRate']
    drgnfly.numEpochs = config['numEpochs']
    drgnfly.numInputNodes = config['numInputNodes']
    drgnfly.numNodesPerHiddenLayer = config['numNodesPerHiddenLayer']
    drgnfly.numOutputNodes = config['numOutputNodes']
    drgnfly.rateChangeBool = config['rateChangeBool']
    drgnfly.testingDataSize = config['testingDataSize']
    drgnfly.trainingDataMax = config['trainingDataMax']
    drgnfly.trainingDataMin = config['trainingDataMin']
    drgnfly.trainingDataSize = config['trainingDataSize']
    drgnfly.numHiddenLayers = config['numHiddenLayers']


def loadModelWeights(filename, model):
    """ Load Weights from param file and add them to the model"""
    model_params = []
    with open(filename, mode ='r')as file:
        csvFile = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)

        for lines in csvFile:
            model_params.append(lines[0])


    counter = 0
    """ Add Weights """
    for l in range(len(model.layers)):
        for n in range(len(model.layers[l].neurons)):
            for w in range(len(model.layers[l].neurons[n].w)):
                model.layers[l].neurons[n].w[w].data = model_params[counter]
                counter += 1

    """ Add Bias Values """
    for l in range(len(model.layers)):
        for n in range(len(model.layers[l].neurons)):
            model.layers[l].neurons[n].b.data = model_params[counter]
            counter += 1



def initalize(args):    
    """ initalize DRGNFLY Object """
    drgnfly = DrgnflyObj(0,0,0,0,0,0,0,0,0,0,0,0,0,0)

    """ Ensure Training Mode is Defined """
    if args.evaluate == None and args.keep_training == None and args.new_model == None:
        logging.warning('Please Define and Execution Mode.')
        sys.exit()

    """ Check for Config File """
    if args.config == None:
        logging.warning('Configuration File Needed')
        sys.exit()
    else:
        logging.warning('Using Model Configuration From: '+  args.config)
        loadConfigFile(args.config, drgnfly)


    """ Load Training Data if Provided"""
    if args.training_data:
        #TODO: Call the New Model Stuff With Input Data
        Data = modelData(drgnfly.numInputNodes, drgnfly.trainingDataMin, drgnfly.trainingDataMax, True, args.training_data)
        #loadTrainingData()
        logging.warning('Using Training Data From: '+  args.training_data)
        #loadTrainingData(args.training_data)
    else:
        #TODO: Call the New Model Stuff Without Input Data
        logging.warning('No Training Data Provided, Will Generate Trainging Data From Config File')
        """ Create Training Data """
        Data = modelData(drgnfly.numInputNodes, drgnfly.trainingDataMin, drgnfly.trainingDataMax, False, args.training_data)
        Data.generateTrainData(drgnfly.trainingDataMin, drgnfly.trainingDataMax, drgnfly.inputFunct)


    """ Initialize Model 
    EX: 
        - 3 Input Nodes
        - 2 Hidden Layers (10 Nodes)
        - 1 Output Node
    Neuron Arch: relu((w1x1 + w2x2) + bias)
    Initilization: 
        model = MLP(3, [10, 10, 1]) 
    """
    model_arch = []
    for i in range(drgnfly.numHiddenLayers):
        model_arch.append(drgnfly.numNodesPerHiddenLayer) 
    model_arch.append(drgnfly.numOutputNodes)

    model = MLP(drgnfly.numInputNodes, model_arch)

    """ Train Model Execution Mode """
    if args.train:
        logging.warning('Execution Mode Set to Train Model')


    if args.parameters != None:
        logging.warning('Using Model Parameters From: '+  args.parameters)
        drgnfly.loadWeights = True
        loadModelWeights(args.parameters, model)
    else:
        logging.warning('No Model Parameters Provided, Training Model with Random Params')
        drgnfly.loadWeights = False


    """ Evaluate Execution Mode """
    if args.evaluate:
        # if args.parameters != None:
        #     #TODO: Call the Model Evaluation Stuff
        logging.warning('DRGNFLY Model Running in Model Evaluation Mode')
        # else:
        #     logging.warning('Please ensure model paramaters and config files have been provided.')
        #     sys.exit()


    return drgnfly, Data, model #, inputData, Evaluation bool?
