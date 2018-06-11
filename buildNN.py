
# 1. random initialize parameters
# 2. forward propagation
    # 2.1 compute one Z
    # 2.2 compute one A (A0 is X)
    # 2.3 loop the two above functions
# 3. compute cost
# 4. compute grads by back propagation
    # 4.1 compute dA
    # 4.2 compute dW and db
    # 4.3 combine them
# 5. update parameters
# 6. compute results (p, not 0/1)

import numpy as np
import matplotlib.pyplot as plt
# ----------------------- estibalish the NN -----------------
def BuildNN(layer_dims, learning_rates, X, Y, iterations = 2000, print_cost = False, threshold = 0.5):

    learning_len = len(learning_rates)
    Yhat = []
    final_params = []
    for j in range(0, learning_len):
        learning_rate = learning_rates[j]
        params = initialize_params(layer_dims)
        print(params)
        costs = []
        for i in range(0, iterations):
                print("iteration: " + str(i))
                AL, caches = forward(X = X, parameters= params)
                cost = compute_cost(AL= AL, Y= Y)
                grads = compute_grad(AL= AL, Y= Y, caches= caches)
                print(grads)
                params = update_parameters(parameters= params, grads= grads, learning_rate= learning_rate)
                print(params)
                if print_cost and i % 1 == 0:
                    print("Cost after iteration %i: %f" %(i, cost))
                if print_cost and i % 1 == 0:
                    costs.append(cost)
                    if i > 0:
                        if costs[i] > costs[i - 1]:
                            final_params.append(params)
                            break;
                if i == iterations - 1:
                    Yhat.append(AL)
                    final_params.append(params)

        plt.plot(np.squeeze(costs))
        plt.ylabel("cost")
        plt.xlabel("iterations (per tens)")
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()

        # predictions = (Yhat[j] > threshold)
        # print("Training Set Accuracy :%d" % float(
        #     (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + "%")

    return final_params

# --------------- prediction ------------------
def test(X, Y, params, threshold = 0.5):
    AL, caches = forward(X=X, parameters=params)
    cost = compute_cost(AL=AL, Y=Y)
    print("cost: " + str(cost))
    predictions = (AL > threshold)
    print("Accuracy :%d" % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1- predictions.T))/float(Y.size)*100) + "%")
    return  predictions

def predict(X, params, threshold = 0.5):
    AL, caches = forward(X=X, parameters=params)
    return  AL
# ----------------------------forward propagation---------------------

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
    A = np.tanh(Z)
    cache = (Z)
    return A, cache

# --------------------------back propagation-------------------

def compute_grad(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L-1]

    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, activation = "sigmoid")


    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l+1)], current_cache, activation= "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def compute_params(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis = 1, keepdims = True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def activation_backward(dA, cache, activation) :
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = compute_params(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = compute_params(dZ, linear_cache)

    return dA_prev, dW, db

def relu_backward(dA, cache):
    A = cache
    dZ = (A > 0)
    return dZ

def tanh_backward(dA, cache):
    A = cache
    dZ = np.dot(dA.T, (1- pow(tanh(A),2)))
    return dZ

def sigmoid_backward(dA, cache):
    A = cache
    dZ = np.dot(dA.T, np.dot(A, (1 - A).T))
    dZ = dZ.T
    return dZ

# --------------------------update params by gradient descent-------------------
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" +str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters