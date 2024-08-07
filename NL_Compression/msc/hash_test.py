import os
import numpy as np
import itertools as it


"""
Simplification Notes:
- All keys with three of the same numbers need to stay as there will only be one instance of those values. 
- Simplification 1: Keys like 510 can be removed becasue we already have 015
- Simplification 2: Keys like 112 and 256 can be adjacent as that can be written as 11256
- Simplification 3: There are three instances of every non-repeating value (one instance of 888 but 3 instances of 203)
"""


def generateTrainData(max):
    x = []
    for i in range(0, max):
        x.append('{0:04b}'.format(i)) #CHANGE!!!
    return(x)


def checkString(string, x):
    string = string# + string[0:len(x[0])]
    correct = 0

    for j in range(len(x)):
        for i in range(len(string)-len(x[0])):
            test_string = string[i:i+4] #CHANGE!!!
            #reversed_test_string = test_string[::-1]
            if test_string == x[j]:# or reversed_test_string == x[j]:
                #print('String: ' + test_string + '  ---  Reversed: ' + reversed_test_string + '  ---  X: ' + x[j])
                correct += 1
                break
    print('Correct = ' + str(correct) + '/' + str(len(x)))
    return correct

    

if __name__ == "__main__":
    x = generateTrainData(16) #CHANGE!!!
    working_string = '1111101001101110010001010100000'
    not_working_string = '111110100110111001000101010000'


    result_correct = checkString(working_string, x)
    
    correct = 0
    string_counter = 1

    while correct != len(x):
        test_string = bin(string_counter)[2:]
        print(test_string)
        correct = checkString(test_string, x)
        string_counter += 1

    



