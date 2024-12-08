import random
import numpy as np
from collections import deque
import math
import time
import ctypes
c_prop_lib = ctypes.CDLL("./propagation.so")
fpointer3d = np.ctypeslib.ndpointer(dtype=np.float64,ndim=3)
fpointer2d = np.ctypeslib.ndpointer(dtype=np.float64,ndim=2)
fpointer1d = np.ctypeslib.ndpointer(dtype=np.float64,ndim=1)
intpointer1d = np.ctypeslib.ndpointer(dtype=np.int32,ndim=1)
c_prop_lib.add_changes_c.argtypes = [fpointer3d,fpointer2d,fpointer2d,intpointer1d,intpointer1d,intpointer1d]
c_prop_lib.add_changes_c.restype = None
#g++ -fPIC -shared -o propagation.so propagation_cplusplus.cpp cplusplus_bindings.cpp

#make test function for this
def add_changes_wc(changes, activations, deltas, model_structure):
    activations_shape = [len(activations),len(activations[0])]
    deltas_shape = [len(deltas),len(deltas[0])]
    
    c_prop_lib.add_changes_c(changes,activations,deltas,model_structure,deltas_shape,activations_shape, len(model_structure))

ch = []
#ch = np.array([], dtype="int32")
a = [0,0]
d = [0,0]
ch_d = deque()
ch_d.append(np.zeros((4,4),dtype=np.float64))
ch_d.append(np.zeros((4,3),dtype=np.float64))
ch_d.append(np.zeros((3,3),dtype=np.float64))
ch_g_test = np.array(ch_d, dtype = object) #np.float64
ch_f_test = np.array(ch_g_test, dtype = np.float64)
model_test = [4,3,3]
d_test = [[3,2],[2,3]]
a_test = [[3,2],[2,3]]
ch.append(np.zeros((4,4)))
ch.append(np.zeros((4,3)))
ch.append(np.zeros((3,3)))
#ch_test = ch_f_test.astype(np.float64)
print(ch_f_test)
#add_changes_wc(3,ch,a,d,model_test,d_test,a_test)
add_changes_wc(ch_f_test,a_test,d_test,model_test)
print(ch_f_test)












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
