# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 14:49:44 2019

@author: peeters Roel
"""

""" 
Masterthesis project: Determining fluid variables with AI
version: 3

This file packs all necessary functions to implement training and utilization of a neural network, 
it is to be imported in a code where training and test examples will be run through to train the network and obtain the parameters.

This code is based on the neural network designed in the Coursera course: Neural Networks and Deep Learning, by Andrew Ng.

Version 3 updates: 
    - tanh and linear activation functions + derivatives
    - updated functionality of DNN-training
    
"""

# importing required packages 

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#####  ACTIVATION FUNCTIONS AND DERIVATIVES #####

def relu(Z):
    """
    Implement the RELU function of Z.
    Input:      Z -- linear layer output
    Output:     A -- post-activation 
                cache -- containing "Z" (for backpropagation)
    """
    A = np.maximum(0,Z) 
    assert(A.shape == Z.shape)
    cache = Z 
    return A, cache

def relu_derivative(dA, cache):
    """
    Implement the derivative for a single RELU unit of dA.
    Input:      dA -- gradient of J to the activation of this layer
                cache -- dictionary from the forward propagation which stored Z
    Output:     dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def leaky_relu(Z):
    """
    Implement the Leaky-RELU function of Z.
    Input:      Z -- linear layer output
    Output:     A -- post-activation 
                cache -- containing "Z" (for backpropagation)
    """
    A = np.maximum(0.01*Z, Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def leaky_relu_derivative(dA, cache):
    """
    Implement the derivative for a single Leaky-RELU unit of dA.
    Input:      dA -- gradient of J to the activation of this layer
                cache -- dictionary from the forward propagation which stored Z
    Output:     dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.where(Z<0, 0.01*dA, dA)
    assert(dZ.shape == Z.shape)
    return dZ

def tanh(Z):
    """
    Implements the TANH activation function
    Input:      Z -- linear layer output
    Output:     A -- post-activation parameter (value between -1,1) 
                cache -- containing "Z" (for backpropagation)
    """
    A = np.tanh(Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def tanh_derivative(dA, cache):
    """
    Implement the derivative of the TANH-activation function.
    Input:      dA -- gradient of J to the activation of this layer
                cache -- dictionary from the forward propagation which stored Z
    Output:     dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    dZ = dA * (1-np.tanh(Z)**2)
    assert(dZ.shape == Z.shape)
    return dZ

def linear(Z):
    """
    Implements a linear activation function
    Input:      Z -- linear layer output
    Output:     A -- post-activation parameter (=Z)
                cache -- containing "Z" (for backpropagation)
    """
    A = deepcopy(Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A, cache

def linear_derivative(dA, cache):
    """
    Implement the derivative of linear activation function
    Input:      dA -- gradient of J to the activation of this layer
                cache -- dictionary from the forward propagation which stored Z
    Output:     dZ -- gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    assert(dZ.shape == Z.shape)
    return dZ


##### INITIALIZATION FUNCTIONS FOR DNN #####
def initialize_parameters(layer_dims):
    """
    Initializes the W and b parameters for each layer in the NN.
    Input:  layer_dims -- amount of neurons in each layer, len(layer_dims) = # of layers in the NN
    Output: parameters -- a dictionary with all the generated parameters W and b for the linear part of a neuron
    """
    parameters = {}
    L = len(layer_dims)            # number of layers in the network
    
    for l in range(1, L):

        parameters['W' + str(l)] = np.random.rand(layer_dims[l], layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
                
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
      
    return parameters

def initialize_parameters_Adam(parameters):
    """ 
    Initializes the extra parameters v and s for use in the Adam optimization algorithm
    Input:  parameters -- the initialized W and b parameters 
    Output: v, s -- dictionaries filled with 0 for use in the Adam optimization
    """
    
    L = len(parameters)//2
    v = {}
    s = {}
    
    for l in range(1,L):
        v["dW"+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        v["db"+str(l)] = np.zeros(parameters['b'+str(l)].shape)
        s["dW"+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        s["db"+str(l)] = np.zeros(parameters['b'+str(l)].shape)
        
    return v, s

##### FORWARD PROPAGATION FUNCTIONS OF A NEURAL NETWORK  ##### 
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Input:      A -- activations of the previous layer
                W, b -- weights and biases for the current layer
    Output:     Z -- linear part of the forward propagation
                cache -- contains current parameters to be used in the backward propagation
    """
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the ACTIVATION layer
    Input:      A_prev -- the activations of the previous layer
                W, b -- weights and biases for the current layer
                activation -- the activation function to be used for this layer
    Output:     A -- activation for the current layer of the NN
                cache -- contains all the parameters to be kept for backward propagation
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'tanh':
        A, activation_cache = tanh(Z)
    elif activation == 'linear':
        A, activation_cache = linear(Z)
    elif activation == 'leaky_relu':
        A, activation_cache = leaky_relu(Z)
    else:
        raise ValueError('No activation has been selected')
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, activation):
    """
    Implement a single full forward propagation pass of NN
    Input:      X -- the input parameters
                parameters -- the initialized parameters W and b
                activation -- the activation function for the final layer
    Output:     AL -- the activations of the final layer L, = to y_hat
                caches -- list of all the caches of the forward propagation
    """
    caches = []                             # will contain all the cached parameters of the forward propagation
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'leaky_relu')
        caches.append(cache)
    
    # Implement final LINEAR ->  ACTIVATION
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation)
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches


##### COST COMPUTATION OF THE AL OF THE CURRENT FORWARD PROPAGATION #####
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation of MSE.
    Input:      AL -- predictions of the final layer
                Y -- actual values of the training/test examples
    Output:     cost -- cost comparisson of AL and Y
    """
    m = Y.shape[1]

    cost = (1./m) * np.sum((Y-AL)**2)
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

##### BACKWARD PROPAGATION PART FOR TRAINING THE NEURAL NETWORK  #####
    
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)
    Input:      dZ -- derivative of cost to the current's layer Z
                cache -- cache containing the parameters of the current layer
    Output:     dA_prev, dW, db -- gradient of cost with regards to A_prev, W and b respectively    
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    Input:      dA -- gradient of the cost to the current layer's activation
                cache -- current layer's parameters from the forward propagation
                activation -- the activation function used in the current layer
    Output:     dA_prev -- gradient of the cost to the previous layer's activation part
                dW, db -- gradient of the cost to the current layer's parameters
    """
    linear_cache, activation_cache = cache
    
    if activation == 'relu':
        dZ = relu_derivative(dA, activation_cache)
    elif activation == 'tanh':
        dZ = tanh_derivative(dA, activation_cache)
    elif activation == 'linear':
        dZ = linear_derivative(dA, activation_cache)
    elif activation == 'leaky_relu':
        dZ = leaky_relu_derivative(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches, activation):
    """
    Implement the full backward propagation of the NN
    Input:      AL -- output activations of the final layer
                Y -- actual values of the training/test examples
                caches -- all the caches stored during the forward propagation
                activation -- the activation of the final layer
    Output:     grads -- dictionary with all the gradients to the cost of the parameter
    """
    grads = {}
    L = len(caches) # the number of layers
    #m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = -2*(Y-AL)
    
    # Lth layer gradients. 
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'leaky_relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


##### FUNCTIONS TO UPDATE THE PARAMETERS OF THE NN #####
    
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    Input:      parameters -- current parameters of the layer
                grads -- current calculated gradients of the layer
                learning_rate -- the given learning rate to update the parameters
    Output:     parameters -- updated parameters W and b for the current layer
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters

def update_parameters_Adam(parameters, v, s, t, grads, learning_rate):
    """
    Update the parameters using Adam optimizer
    beta1, beta2 and epsilon are hardcoded
    Input:      parameters, v, s -- current parameters of the layer
                t -- epoch number
                grads -- current gradients of the layer
                learning_rate -- the given learning rate to update the parameters
    Output:     parameters, v, s -- updated parameters W and b and v,s for the current layer
    """
    
    L = len(parameters)//2
    v_corr = {}
    s_corr = {}
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    for l in range(1,L):
        v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*grads["dW" + str(l)]
        v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*grads["db" + str(l)]
        
        v_corr["dW" + str(l)] = v["dW" + str(l)]/(1-beta1**t)
        v_corr["db" + str(l)] = v["db" + str(l)]/(1-beta1**t)
        
        s["dW" + str(l)] = beta2*s["dW"+str(l)] + (1-beta2)*(grads["dW"+str(l)]**2)
        s["db" + str(l)] = beta2*s["db"+str(l)] + (1-beta2)*(grads["db"+str(l)]**2)
        
        s_corr["dW" + str(l)] = s["dW"+str(l)]/(1-beta2**t)
        s_corr["db" + str(l)] = s["db"+str(l)]/(1-beta2**t)
        
        parameters["W" + str(l)] = parameters["W"+str(l)] - learning_rate*v_corr["dW"+str(l)]/(np.sqrt(s_corr["dW"+str(l)])+epsilon)
        parameters["b" + str(l)] = parameters["b"+str(l)] - learning_rate*v_corr["db"+str(l)]/(np.sqrt(s_corr["db"+str(l)])+epsilon)
        
    return parameters, v, s   
    

##### FULL IMPLEMENTATION FOR TRAINING A NEURAL NETWORK WITH NUMBER OF LAYERS AS SPECIFIED #####

def L_layer_model(X, Y, layers_dims, activation, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, optimizer = 'GD'):
    """
    Implements a L-layer neural network.
    Input:      X -- input features of the training examples
                Y -- output values of the training examples
                layers_dims -- dimensions of the layers for the network
                activation -- activation function of the final layer
                learning_rate -- learning rate for the update of the parameters
                num_iterations -- number of iterations the NN will train itself
                print_cost -- boolean to determine whether to print costs
                optimizer -- optimizer to be used for the update of the parameters
    Output:     parameters -- trained parameters
                grads -- final gradients of the neural network
                costs -- stored costs of the neural network
    """
    costs = []                         # keep track of cost
    t = 0
    
    # Factor to be used in printing the cost, depending on the number of iterations
    if num_iterations <= 10000:
        print_factor = 10
    elif num_iterations <= 50000:
        print_factor = 50
    elif num_iterations <= 100000:
        print_factor = 100
    elif num_iterations <= 500000:
        print_factor = 500
    elif num_iterations <= 1000000:
        print_factor = 1000
    else:
        print_factor = 5000
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters(layers_dims)
    
    if optimizer == 'Adam':
        v, s = initialize_parameters_Adam(parameters)
       
    # Loop over the number of iterations to perform gradient descent
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> FINAL ACTIVATION
        AL, caches = L_model_forward(X, parameters, activation)
        
        # Compute cost
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches, activation)
 
        # Update parameters
        if optimizer == 'GD':
            parameters = update_parameters(parameters, grads, learning_rate)
        elif optimizer == 'Adam':
            t += 1
            parameters, v, s = update_parameters_Adam(parameters, v, s, t, grads, learning_rate)
    
        # Print the cost every X training example
        if print_cost and i % (num_iterations/print_factor) == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if i % print_factor == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.yscale('log')
    plt.ylabel('cost')
    plt.xlabel('iterations' + ' (x ' + str(print_factor) +')')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters, grads, costs

# Implementing a function to predict the resultant Force given a 2D-velocity field
# The result will be evaluated versus the actual force given in the dev- or testset

def predict_test(X, y, parameters, activation):
    """
    This function is used to predict the results of a the L-layer neural network.
    A comparison with the correct results is made and the accuracy is determined.
    """
    
    m = X.shape[1]
    # n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    p, caches = L_model_forward(X, parameters, activation)
    error = p-y
    accuracy = np.sqrt(np.mean(error**2))/np.sqrt(np.mean(y**2))
    #deviation = np.std(spread)
 
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    #print("Accuracy: "  + str(accuracy))
    #print("Deviation: " + str(deviation))
        
    return p, error, accuracy

# Implementing the function to start a prediction for a new data set of velocity field
    
def predict(X, parameters):
    """
    This function is used to predict the results of a the L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- prediction for the given dataset X
    """
    
    m = X.shape[1] 
    # n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    p, caches = L_model_forward(X, parameters)
 
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    #print("Predicted Force magnitude: "  + str(p) + " in Newton")
        
    return p

##### Functions to perform gradient checking of a parameter calculated in the NN #####
    
def gradient_check(X, Y, parameters, parameter, node, gradients, activation):
    """ 
    This function checks the W or b in a node in the trained NN
    X, Y -- training data
    parameters -- the trained parameters
    parameter -- the parameter to be checked, must be a string
    node -- the node in the layer to be checked (must be a tuple)
    gradients -- the gradients of the NN
    
    output : difference = ||grads - gradapprox||/(||grads||+||gradapprox||)
    """
    
    epsilon = 1e-7
    
    thetaplus = deepcopy(parameters)
    thetaplus[parameter][node] += epsilon
    Yhat_plus,_ = L_model_forward(X, thetaplus, activation)
    J_plus = compute_cost(Yhat_plus, Y)
    thetamin = deepcopy(parameters)
    thetamin[parameter][node] -= epsilon
    Yhat_min,_ = L_model_forward(X, thetamin, activation)
    J_min = compute_cost(Yhat_min, Y)

    gradapprox = (J_plus - J_min) / (2*epsilon)

    grads = gradients['d'+parameter][node]

    numerator = np.linalg.norm((grads - gradapprox))
    denominator = np.linalg.norm(gradapprox)+np.linalg.norm(grads)
    diff = numerator/denominator
    
    return diff 


##### Function to transform the gradients dictionary to a vector #####
# vector required when checking the gradients
def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in gradients:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta