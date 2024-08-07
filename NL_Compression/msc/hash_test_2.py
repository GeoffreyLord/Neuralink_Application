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
    for j in range(0,max):
        for i in range(0, max):
            temp = [j, i]
            x.append(temp) #CHANGE!!!
    return(x)


def checkString(string, x):
    string = str(string) + str(string)[0:len(x[0])]
    correct = 0

    for j in range(len(x)):
        x_str = str(x[j][0]) + str(x[j][1])
        for i in range(len(string)-len(x_str)):
            test_string = str(string)[i:i+len(x_str)]
            reversed_test_string = str(test_string)[::-1]
            if test_string == x_str or reversed_test_string == x_str:
                #print('String: ' + test_string + '  ---  Reversed: ' + reversed_test_string + '  ---  X: ' + x_str)
                correct += 1
                break
    print('Correct = ' + str(correct) + '/' + str(len(x)))
    return correct

    

if __name__ == "__main__":
    x = generateTrainData(10) #CHANGE!!!
    print(x)
    
    correct = 0
    string_counter = 100

    while correct != len(x):
        test_string = string_counter
        print(test_string)
        correct = checkString(test_string, x)
        string_counter += 1

    



