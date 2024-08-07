from management.viz import plotWeights
import matplotlib as plt
import statistics
import numpy as np
import logging
import math
import sys


"""
Notes: As discussed in the updated whitepaper. There is a need to profile models based on the plots that were generated during profiling. It turns
out that these models all have distinct characteristics based on their lines of best fits. We need to generate an algo that allows for ligns to be characterized wtihout
being too spesific about their exact points as then we will miss matchning. What we can do is create a way to charatersze these lines and then give a score to each models
between 1-100 noting how they conform to the final value. 

The issue is I need to figure out how to compare these lines

"""

#TODO: This needs to maybe check for model accuracy.
""" profileModel attempts to identify the foundationa mathmatical operands affecting each input variable"""
def profileModel(drgnfly, inputs, weight_arr, bias_arr):
    logging.warning('====================Profiling Model====================')
    """ Check the Mean and Standard Deviations """
    weight_mean =  np.zeros((drgnfly.numInputNodes))
    weight_stdev = np.zeros((drgnfly.numInputNodes)) 

    bias_mean = statistics.mean(bias_arr)
    bias_stdev = statistics.stdev(bias_arr)

    for i in range(drgnfly.numInputNodes):
        weight_mean[i] = statistics.mean(weight_arr[:,i])
        weight_stdev[i] = statistics.stdev(weight_arr[:,i])


    """ Determine Mathmatical Function For Inputs """
    #TODO: This needs some more thought generally. 

    """
        Knowns:
            -[x] If a weight has a value with low standard deviation we know that weight is likley multiplied or devided by a constant.
            -[] If a weight has a value with high standard deviations we know that weight is multiplied or devided by a dynamic parameter
                or the weight is raised to a power. 
            -[x] If the bias has a low standard deviation it is likley that a constant is added to a parameter. 
            -[] If the bias has a high standard deviation then it is used to accomidate for a more complex mathmatical operation against the weights 
                or one of the input values is added to the variable multiple times. 


        Questions:
            - What type of patterns can be seen with sin, cos, and tan?
            - What does a's weight look like when multiple instances of it show up in the equation?
            - What if input values are not predictable in nature. Meaining, there are no instances of input paramaters of one or zero. Moreover,
                if instances of one or zero are outside of a range where the model accuralty operates then you can not expect to use those values for testing
                and get accurate results. Therefore, we need to find a way to identify how you can use two random input values and identify the weight
                relationship between them. Could the slope of the weight change work?
                    - The slope of the weights if a value that is raised to the power of two would be positive.

    
    """


    """ Preallocate Vars """
    analysis_stdev_threshold = 1
    analysis_slope_threshold = 0.2
    constant_set_threshold = 0.1
    assumptions = np.zeros((drgnfly.numInputNodes + 1))
    assumptions_flag = np.zeros((drgnfly.numInputNodes + 1)) #Flag for if an assumption is made
    common_constants = [0.00, 0.33, 0.50, 0.66, 0.75, 1.00, 2.00, 2.72, 3.00, 3.14, 4, 5, 6, 7, 8, 9, 10, -0.33, -0.50, -0.66, -0.75, -1.00, -2.00, -2.72, -3.00, -3.14, -4, -5, -6, -7, -8, -9, -10]
    input_slopes = np.zeros((drgnfly.numInputNodes))


    """ Calculate Slopes of Weight vs inputs best fit line """
    for i in range(drgnfly.numInputNodes):
        unique_inputs = np.unique(inputs.X[:,i])
        input_polyfit = np.poly1d(np.polyfit(inputs.X[:,i], weight_arr[:,i], 1))(np.unique(inputs.X[:,i]))
        input_slopes[i] = (input_polyfit[-1] - input_polyfit[0]) / (unique_inputs[-1] - unique_inputs[0])
    print(input_slopes)


    """ Identify if any input value weights are constant multiplicatives """
    for i in range(drgnfly.numInputNodes):
        logging.warning('==========Profiling Paramater #' + str(i+1) + '=========')
        if weight_stdev[i] < analysis_stdev_threshold:
            logging.warning('Parameter #' + str(i+1) + ' is likley multiplied by a constant value of: ' + str(weight_mean[i]) + 
                            ' or this value is converging to a near integer like: ' + str(np.round(weight_mean[i],1)) + '.') 
            for j in range(len(common_constants)):
                if (weight_mean[i] < common_constants[j]+constant_set_threshold) and (weight_mean[i] > common_constants[j]-constant_set_threshold):
                    logging.warning('Assuming parameter #'  + str(i+1) + ' is likley a constant value of ' + str(common_constants[j]) + '.')
                    assumptions[i] = common_constants[j]
                    assumptions_flag[i] = 1
        elif assumptions_flag[i] == 0:
            logging.warning('Parameter #' + str(i+1) + ' may be raised to the power of ' + str(np.round(input_slopes[0])) + '.')
        else:
            logging.warning('Parameter #' + str(i+1) + ' does not seem to be multiplied by a constant value or model accuracy is not high enough.')


    """ Identify if the bias is constant """
    logging.warning('==========Profiling Model Bias=========')
    if bias_stdev < analysis_stdev_threshold:
        logging.warning('Model Bias is likley a constant value of: ' + str(bias_mean) + ' or this value is converging to a near integer like: ' + str(np.round(bias_mean)) + '.') 
        for i in range(len(common_constants)):
            if (bias_mean < common_constants[i]+constant_set_threshold) and (bias_mean > common_constants[i]-constant_set_threshold):
                    logging.warning('Assuming model bias is likley a constant value of ' + str(common_constants[i]) + '.')
                    assumptions[-1] = common_constants[i]
                    assumptions_flag[-1] = 1
    else:
        logging.warning('The model bias does not seem to be a constant value or model accuracy is not high enough.')

    """ Create Subplots for each input vaiable """
    plotWeights(inputs, weight_arr, drgnfly, bias_arr)

    """ If assumptions have been made for all values generate function prediction 
            This can attirbute functions with only constant weights like Y = 2a+3b+0c.
    """
    if (np.sum(assumptions_flag) == len(assumptions_flag)):
        prediction_str = 'Y = '
        logging.warning('==========Possible Function=========')
        for i in range(drgnfly.numInputNodes):
            prediction_str += '(var_' + str(i) + ' * ' + str(assumptions[i]) + ') + '
        prediction_str += str(assumptions[-1])
        logging.warning(prediction_str)
        sys.exit()




    """ Take assumptions and calculate bias error based on assumptions. """

    # if drgnfly.numInputNodes == 3:
    #     #if weight_stdev[i] < 0.2:
    #     logging.warning('==========Possible Functions=========')
    #     logging.warning('Y = ' + str(weight_mean[0]) + '*a + ' + str(weight_mean[1]) + '*b + ' + str(weight_mean[2]) + '*c + ' + str(bias_mean))
    #     logging.warning('Y = ' + str(weight_mean[0] + bias_mean) + '*a + ' + str(weight_mean[1]) + '*b + ' + str(weight_mean[2]) + '*c')
    #     logging.warning('Y = ' + str(weight_mean[0]) + '*a + ' + str(weight_mean[1] + bias_mean) + '*b + ' + str(weight_mean[2]) + '*c')
    #     logging.warning('Y = ' + str(weight_mean[0]) + '*a + ' + str(weight_mean[1]) + '*b + ' + str(weight_mean[2] + bias_mean) + '*c')
    #     logging.warning('Y = ' + str(np.round(weight_mean[0])) + '*a + ' + str(np.round(weight_mean[1])) + '*b + ' + str(np.round(weight_mean[2])) + '*c + ' + str(np.round(bias_mean)))


    """ Determine Bias Predicted Value """

