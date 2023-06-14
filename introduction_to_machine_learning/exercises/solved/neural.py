import matplotlib.pyplot as plt
import numpy as np


###########################################################################
# Implement the relu_activation, softmax_activation, forward_propagation, #
# calculate_loss, backpropagation functions such that all tests pass.     #
###########################################################################


# The activation function used for intermediate layers.
# Z: the linear combination evaluation of a layer without the activation function, np.narray (n, m)
# returns the activation of the layer, np.narray(n, m)
#   where n is the number of neurons in the layer
#         m is the number of samples in the batch
def relu_activation(Z):
  A = np.copy(Z)
  A[A < 0.0] = 0.0
  return A


# We need to classify samples in the data set therefore the activation of the last layer must be a 
# probability distribution. This will be our prediction.
# Z: the linear combination evaluation of the last layer without the activation function, np.narray (n, m)
# returns the activation of the layer, np.narray(n, m)
#   where n is the number of neurons in the layer
#         m is the number of samples in the batch
def softmax_activation(Z):
  A = np.exp(Z)
  A = A / np.sum(A, axis=1)
  return A


# The output of the neural network WITHOUT the last layer activation.
# For intermediate layers use the relu_activation.
# Used to draw the decision boundary of the model and to calculate the full forward propagation.
# Ws: a list of weight matrices for each layer in the neural network, list of np.narray (ni, nj)
# Bs: a list of bias vectors for each layer in the neural network, list of np.narray (ni, 1)
# X: a batch of features from the data set, np.narray (n0, m)
# returns the activation of each layer, list of np.narray (ni, m)
#   where n0 is the number of features
#         ni is the number of neurons for the layer i
#         nj is the number of neurons for the layer i - 1
#         m is the number of samples in the batch
def forward_propagation_without_last_activation(Ws, Bs, X):
  A = X
  As = []
  for i in range(len(Ws)):
    A = Ws[i] @ A + Bs[i]
    if i + 1 != len(Ws):
      A = relu_activation(A) 
    As.append(A)


# The output of the neural network.
# For intermediate layers use the relu_activation and for the last layer use the softmax_activation.
# Ws: a list of weight matrices for each layer in the neural network, list of np.narray (ni, nj)
# Bs: a list of bias vectors for each layer in the neural network, list of np.narray (ni, 1)
# X: a batch of features from the data set, np.narray (n0, m)
# returns the activation of each layer, list of np.narray (ni, m)
#   where n0 is the number of features
#         ni is the number of neurons for the layer i
#         nj is the number of neurons for the layer i - 1
#         m is the number of samples in the batch
def forward_propagation(Ws, Bs, X):
  As = forward_propagation_without_last_activation(Ws, Bs, X)
  As[-1] = softmax_activation(As[-1])
  return As


def calculate_error(A, Y):
  return -(1 - Y) * np.log(1 - A) - Y * np.log(A)


# The loss for the current batch.
# A: the activation of the last layer, np.narray (n, m)
# Y: the ground truth, np.narray (n, m)
# returns the activation of each layer, list of np.narray (ni, m)
#   where n is the number of neurons for the last layer
#         m is the number of samples in the batch
def calculate_loss(E):
  return np.mean(np.sum(E, axis=1))



def backpropagation(Ws, Bs, As, Y):
  dWs_dloss = []
  dBs_dloss = []
  return dWs_dloss, dBs_dloss


##############################################################################
# The code below should not be changed prior to completing the exercises.    #
# After you pass all the tests we encourage you to play with the code below. #
##############################################################################


def read_the_data_set():
  with open('../../data/hard_processes.txt', 'r') as f:
    data = np.array([[float(val) for val in line.strip().split(' ')] for line in f.readlines()]).T
    X = data[0:2,:].reshape(2, -1)
    Y = data[2,:].reshape(1, -1)
  return (X, Y)


def draw_boundry(Ws, Bs, boundary_X, axs, boundary_graph, samples):
  Z = np.zeros((samples, samples))
  for i in range(samples):
    for j in range(samples):
      x1, x2 = boundary_X[0,i], boundary_X[0,j]
      X = np.array([x1, x2]).reshape(-1, 1)
      Z[i,j] = forward_propagation_without_last_activation(Ws, Bs, X)
  if boundary_graph != None:
    for tp in boundary_graph.collections:
      tp.remove()
  boundary_graph = axs[0].contour(boundary_X[0,:], boundary_X[1,:], Z, levels=[0], colors='blue')
  return boundary_graph


def draw_loss(iteration, loss, iterations_axis, losses_axis, axs, loss_graph):
  iterations_axis.append(iteration + 1)
  losses_axis.append(loss)
  loss_graph.set_data(iterations_axis, losses_axis)
  axs[1].set_title(f"Loss: {loss:0.2f}")
  axs[1].relim() 
  axs[1].autoscale_view(True,True,True)


def initialize_graphs(Ws, Bs, X, Y, boundry_samples):
  min_x = np.amin(X, axis=1)
  max_x = np.amax(X, axis=1)
  
  padding = 0.1
  min_x_padded = min_x * (1 - padding)
  max_x_padded = max_x * (1 + padding)

  plt.ion()
  fig, axs = plt.subplots(1, 2)

  axs[0].set_xlim((min_x_padded[0], max_x_padded[0]))
  axs[0].set_ylim((min_x_padded[1], max_x_padded[1]))
  axs[0].scatter(X[0,Y[0]==1], X[1,Y[0]==1], marker='x', color='red', linewidth=1)
  axs[0].scatter(X[0,Y[0]==0], X[1,Y[0]==0], marker='o', color='green', fc='none', ec='green', linewidth=1, s=60)
  axs[0].set_title('Malicious process detection')
  axs[0].set_xlabel('X1 (normalized average written bytes per second)')
  axs[0].set_ylabel('X2 (normalized average erased bytes per second)')

  loss_graph, = axs[1].plot([], [], color='red')
  axs[1].set_xlabel('Iterations')
  axs[1].set_ylabel('Loss')

  boundary_X = np.linspace(min_x_padded, max_x_padded, boundry_samples).T

  boundary_graph = draw_boundry(Ws, Bs, boundary_X, axs, None, boundry_samples)

  return fig, axs, boundary_graph, loss_graph, boundary_X
