# Perceptron implementation without numpy
# wi = wi + learning_rate*(yi+y_pred)*xi
# bi = bi + learning_rate*(yi+y_pred)
# y = w_1*x_1 + w_2*x_2 + b


# Imports
from random import random


# Data
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]
dimensions = len(inputs[0])
w = [2*random() - 1 for _ in range(dimensions)]
b = 2*random() - 1
learning_rate = 1.0


# Support functions
def activation_function(y_pred: float):
    return 1 if y_pred > 0 else 0    

def addictive_function(input_0, input_1):
    return input_0*w[0] + input_1*w[1] + b


# Learn proccess
while True:
    step = 1
    cost = 0
    for input, output in zip(inputs, outputs):
        y_pred = addictive_function(input[0], input[1])
        y_pred_adjusted = activation_function(y_pred)
        error = output - y_pred_adjusted
        
        w[0] = w[0] + learning_rate * error * input[0]
        w[1] = w[1] + learning_rate * error * input[1]
        b = b + learning_rate * error
        
        cost += error**2
        
    if cost == 0:
        print("Steps needed:", step)
        break
    
    step += 1
        
    
# Results, weights and bias calibrated
print("\nWeights and bias:")
print("w_1: ", w[0])
print("w_2: ", w[1])
print("b: ", b)


# Testing
print("\nTests:")
for input in inputs:
    y = input[0]*w[0] + input[1]*w[1] + b
    print("Input:", input, "-", "Output:", activation_function(y))
