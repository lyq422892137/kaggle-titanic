#
# 1. random initialize parameters
# 2. forward propagation
# 2.1 compute one Z
# 2.2 compute one A (A0 is X)
# 2.3 loop the two above functions
# 3. compute cost
# 4. compute grads
# 5. backward propagation
# 6. update parameters
# 7. compute results (p, not 0/1)

import numpy as np

def BuildNN(layer_dims, learning_rate, X, Y):

    params = initialize_params(layer_dims)
    # caches = forward(X, params)
    # print(caches)
    # print(X.shape)
    # A, cache1 = compute_A(X, params["W1"], params["b1"], "relu")
    # print(A.shape)
    # AL, cache2 = compute_A(A, params["W2"], params["b2"], "sigmoid")
    # print(AL.T.shape)
    # print(AL)

    AL, caches = forward(X, params)
    #print(AL.shape)
    cost = compute_cost(AL, Y)
    print(cost)



def initialize_params(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1,L):
        parameters["W" + str(l)] = np.random.rand(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l],1))

        assert (parameters["W" + str(l)].shape == (layer_dims[l],layer_dims[l-1]))
        assert (parameters["b" + str(l)].shape == (layer_dims[l], 1))
    return parameters


def forward(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2

    for l in range(1, L):

        A_prev = A
        A, cache = compute_A(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
    AL, cache = compute_A(A, parameters["W" + str(L)], parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -np.sum(Y* np.log(AL) +(1-Y) * np.log(1-AL))/m
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost

def compute_Z(A,W,b):
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache

def compute_A(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = compute_Z(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = compute_Z(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "tanh":
        Z, linear_cache = compute_Z(A_prev, W, b)
        A, activation_cache = tanh(Z)

    cache = (linear_cache, activation_cache)
    assert (A.shape == (W.shape[0],A_prev.shape[1]))

    return A,cache


def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    cache = (Z)
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    cache = (Z)
    return A, cache

def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z)/np.exp(Z) + np.exp(-Z))
    cache = (Z)
    return A, cache