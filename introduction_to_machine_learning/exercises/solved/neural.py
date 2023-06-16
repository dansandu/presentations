import matplotlib.pyplot as plt
import numpy as np


def sigmoid(Z):
  return 1.0 / (np.exp(-Z) + 1.0)


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
  A[A < 0.0] = A[A < 0.0] * 0.1
  return A


# We need to classify samples in the data set therefore the activation of the last layer must be a 
# probability distribution. This will be our prediction.
# Z: the linear combination evaluation of the last layer without the activation function, np.narray (n, m)
# returns the activation of the layer, np.narray(n, m)
#   where n is the number of neurons in the layer
#         m is the number of samples in the batch
def softmax_activation(Z):
  A = np.exp(Z)
  A = A / np.sum(A, axis=0)
  return A


# The output of the neural network.
# For intermediate layers use the relu_activation function.
# For the last layer use the softmax_activation function for multiple outputs. For single outputs use sigmoid.
# Used to draw the decision boundary of the model and to calculate the full forward propagation.
# Ws: a list of weight matrices for each layer in the neural network, list of np.narray (ni, nj)
# Bs: a list of bias vectors for each layer in the neural network, list of np.narray (ni, 1)
# X: a batch of features from the data set, np.narray (n0, m)
# returns the Z of each layer and the Y_hat prediction, tuple (list of np.narray (ni, m), np.narray (nl, m))
#   where n0 is the number of features
#         ni is the number of neurons for the layer i
#         nj is the number of neurons for the layer i - 1
#         nl is the number of neurons for the last layers also the output layer
#         m is the number of samples in the batch
def forward_propagation(Ws, Bs, X):
  layers = len(Ws)
  Zs = [None] * layers
  A = X
  for i in range(layers):
    Zs[i] = Ws[i] @ A + Bs[i]
    A = sigmoid(Zs[i])
  if Zs[-1].shape[0] == 1:
    Y_hat = A
  else:
    Y_hat = softmax_activation(Zs[-1])
  return Zs, Y_hat


# The loss for the current batch.
# A: the activation of the last layer, np.narray (n, m)
# Y: the ground truth, np.narray (n, m)
# returns the activation of each layer, list of np.narray (ni, m)
#   where n is the number of neurons for the last layer
#         m is the number of samples in the batch
def calculate_loss(Y_hat, Y):
  error = (1 - Y) * np.log(1 - Y_hat) + Y * np.log(Y_hat)
  return -np.mean(np.sum(error, axis=0))


def backpropagation(Ws, Bs, X, Zs, Y_hat, Y):
  layers = len(Ws)

  dloss_dWs = [None] * layers
  dloss_dBs = [None] * layers

  m = X.shape[1]
  
  dloss_dZ = Y_hat - Y
  
  for i in reversed(range(layers)):
    Ap = sigmoid(Zs[i-1]) if i > 0 else X

    dloss_dW = dloss_dZ @ Ap.T / m
    dloss_dB = np.mean(dloss_dZ, axis=1, keepdims=True)

    dloss_dWs[i] = dloss_dW
    dloss_dBs[i] = dloss_dB

    if i != 0:
      dAp_dZp = Ap * (1 - Ap)

      dZ_dAp = Ws[i].T

      dloss_dZ = dAp_dZp * (dZ_dAp @ dloss_dZ)

  return dloss_dWs, dloss_dBs


##############################################################################
# The code below should not be changed prior to completing the exercises.    #
# After you pass all the tests we encourage you to play with the code below. #
##############################################################################


def read_the_data_set():
  with open('../../data/processes2.txt', 'r') as f:
    data = np.array([[float(val) for val in line.strip().split(' ')] for line in f.readlines()]).T
    X = data[0:2,:].reshape(2, -1)
    Y = data[2,:].reshape(1, -1)
  return (X, Y)


def draw_boundry(Ws, Bs, boundary_X, axs, boundary_graph, samples):
  Z = np.zeros((samples, samples))
  for i in range(samples):
    for j in range(samples):
      x1, x2 = boundary_X[0,i], boundary_X[0,j]
      X = np.array([x1, x2]).reshape(2, 1)
      Zs, _ = forward_propagation(Ws, Bs, X)
      Z[i,j] = Zs[-1]
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


def normalize(X):
  Minimum = np.amin(X, axis=1, keepdims=True)
  Maximum = np.amax(X, axis=1, keepdims=True)
  return (X - 0.5 * (Minimum + Maximum)) / (0.5 * (Maximum - Minimum))


def run_gradient_descent(X, Y, initial_Ws, initial_Bs, learning_rate, iterations):
  # fixes the exploding gradient problem and speeds up learning
  Xn = normalize(X)

  Ws = initial_Ws
  Bs = initial_Bs

  losses_axis = []
  iterations_axis = []

  boundry_samples = 50
  fig, axs, boundary_graph, loss_graph, boundary_X = initialize_graphs(Ws, Bs, Xn, Y, boundry_samples)

  for iteration in range(iterations):
    Zs, Y_hat = forward_propagation(Ws, Bs, Xn)

    dloss_dWs, dloss_dBs = backpropagation(Ws, Bs, Xn, Zs, Y_hat, Y)

    for i in range(len(Ws)):
      Ws[i] = Ws[i] - learning_rate * dloss_dWs[i]
      Bs[i] = Bs[i] - learning_rate * dloss_dBs[i]

    # update the graphs every 1000 iterations
    if iteration % 1000 == 0:
      boundary_graph = draw_boundry(Ws, Bs, boundary_X, axs, boundary_graph, boundry_samples)

      loss = calculate_loss(Y_hat, Y)

      draw_loss(iteration, loss, iterations_axis, losses_axis, axs, loss_graph)

      fig.canvas.draw()
      fig.canvas.flush_events()
      plt.pause(1)


def run_tests():
  pass


if __name__ == '__main__':
  X, Y = read_the_data_set()

  # make sure the tests pass before running gradient descent
  run_tests()

  # Build a 3 layers neural network:
  # - 1st layer has 3 neurons
  # - 2nd layer has 2 neurons
  # - 3rd layer has 1 neuron and is the output

  # we can randomly initialize to try different values
  #Ws = [
  #  np.random.rand(3, 2),
  #  np.random.rand(2, 3),
  #  np.random.rand(1, 2),
  #]

  #Bs = [
  #  np.random.rand(3, 1),
  #  np.random.rand(2, 1),
  #  np.random.rand(1, 1),
  #]

  # for debugging purposes you don't want random behaviour
  Ws = [
    np.array([[0.36, 0.28],
              [0.08, 0.81],
              [0.51, 0.17]]),
    np.array([[0.96, 0.00, 0.11],
              [0.38, 0.17, 0.73]]),
    np.array([[0.44, 0.48]]),
  ]

  Bs = [
    np.array([[0.81], 
              [0.79],
              [0.49]]),
    np.array([[0.61],
              [0.37]]),
    np.array([[0.28]]),
  ]

  # the learning rate should be low enough otherwise the algorithm will not converge
  # but increasing the learning rate can increase the convergence speed
  learning_rate = 0.2

  iterations = 40000

  run_gradient_descent(X, Y, Ws, Bs, learning_rate, iterations)

  print("Press enter to exit...", end='')
  input()
