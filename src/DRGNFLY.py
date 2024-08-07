"""
Created By: Geoffrey Lord
Email: Geoffrey.Lord.Jr@gmail.com
Descripton: 
    DRGNFLY is a neural network simplificaiton tool that can both train and profile neural network models. It is build on top of micrograd and is primarily
    catered towards researchers looking to better understand expermental data. 

    I believe neural networks are here to provide more than solutions. Lets help them provide us with insights. 

    Welcome to the DRGNFLY Project!

    DRGNFLY: https://github.com/GeoffreyLord/DRGNFLY
    Bult with micrograd: https://github.com/karpathy/micrograd/tree/master

Usage:
    Train Model Mode (With Input Data):
    ~$ python3 DRGNFLY.py -t -c <model-config.yaml> -d <training-data.csv> 

    New Model Mode (With Input Function):
    ~$ python3 DRGNFLY.py -t -c <model-config.yaml>   

    Proceed Training Model Mode (With Input Data):
    ~$ python3 DRGNFLY.py -t -c <model-config.yaml> -p <model-parameters.csv> -d <training-data.csv> 

    Proceed Training Model Mode (With Input Function):
    ~$ python3 DRGNFLY.py -t -c <model-config.yaml> -p <model-parameters.csv> 

    Evaluate Model Mode (With Input Data Data):
    ~$ python3 DRGNFLY.py -e -c <model-config.yaml> -p <model-parameters.csv> -d <training-data.csv>  

"""

import sys
import random
import logging
import argparse
import numpy as np
import management.initalize as setup
from testing.testModel import evaluateModel
from management.dataMgmt import exportWeights
from management.model import trainModel, testModel
#import matplotlib.pyplot as plt
#from micrograd.nn import Neuron, Layer, MLP
#from micrograd.engine import Value
#from management.dataMgmt import modelData, exportWeights


""" Set Random Seed """
np.random.seed(42) 
random.seed(42) 

""" Set Recusrion Depth """
sys.setrecursionlimit(5000)


if __name__ == "__main__":

    """ Command Line Argument Management """
    parser = argparse.ArgumentParser(prog='DRGNFLY.py', description='Neural Network Scientific Toolkit')
    parser.add_argument('-c', '--config') 
    parser.add_argument('-p', '--parameters')
    parser.add_argument('-d', '--training_data')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    args = parser.parse_args()

    """ Logging Config """
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s', datefmt='%I:%M:%S %p')

   
    """ Determine Model Execution Mode and Load Configuration"""
    drgnfly, data, model = setup.initalize(args)

    if args.train: 
         trainModel(drgnfly, data, model)
         exportWeights(model, drgnfly)


    """ After Training or before testing model accuracy is tested"""
    testModel(drgnfly, data, model)


    if args.evaluate:
        logging.warning('====================Evaluating Model====================')
        if drgnfly.numHiddenLayers < 6:
            evaluateModel(drgnfly, data, model)
        else:
            logging.warning('DRGNFLY Currently Only Supports Analysis of Models with less than 6 hidden layers')



   




        