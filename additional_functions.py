import random
import numpy as np
from collections import deque
import math
import time




EPSILON = 0.00001 #small number to avoid divsion by zero
#adds elements of an array element by element, 

"""
thanks to this webstie for the timing function:
https://builtin.com/articles/timing-functions-python#:~:text=Decorator%20to%20Time%20Functions%20Explained,took%20to%20run%20that%20function.

"""
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper






@timeit
def add_arrays_by_element_3D(ar1,ar2):   #!biggest time lost 
    ar1dim = len(ar1)
    ar2dim = len(ar2)
    if ar1dim != ar2dim:
        print("Error! two arrays or matrix's passed in do not have the same dim")
        exit()
    newWeights = []
    for x in range(len(ar1)): #for each matrix in the lists
        newWeights.append(ar1[x] + ar2[x]) #adds the two matrix's and then appends it to the new weights list
    return newWeights


def sigmoid(input):
    input = round(input,4)
    try:
        output = (((1/(1+(math.e**(-input)))) + EPSILON))
        #print(output)
    except:
        print("ERROR ERROR here is the input:")
        print(input)
    return input
    
def sigmoid_vec(input_vec):  #!! HAVING A OVERFLOW ERROR HERE, SMALLER EPSILON??? still having it, 
    temp_vec = []
    for x in input_vec:
        temp_vec.append(sigmoid(x))
    return temp_vec

def sigmoid_derivative(input):
    return sigmoid(input) * (1-sigmoid(input))

def softmax(vec): 
    outputVec = []
    sum = 0
    for x in vec:
        for y in vec:
            sum += math.exp(y)
        outputVec.append((math.exp(x))/sum)
        print("done with y")
    print("done with x")
    return outputVec
