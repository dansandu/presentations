import matplotlib.pyplot as plt
import numpy as np


##################################################################
# Implement the sigmoid, forward_propagation, calculate_loss and #
# backpropagation functions such that all tests pass.            #
##################################################################


# The activation function for each neuron.
# Z: the linear combination evaluation of a layer without the activation function, np.narray (n, m)
# returns the activation of the layer, np.narray(n, m)
#   where n is the number of neurons in the layer
#         m is the number of samples in the batch
def sigmoid(Z):
  # YOUR CODE HERE #
  return 1.0 / (np.exp(-Z) + 1.0)


# The output of the neural network. Use the sigmoid activation function for all neurons.
# Used to draw the decision boundary of the model and to calculate the forward propagation.
# Ws: a list of weight matrices for each layer in the neural network, list of np.narray (ni, nj)
# Bs: a list of bias vectors for each layer in the neural network, list of np.narray (ni, 1)
# X: a batch of features from the data set, np.narray (n0, m)
# returns the linear combination Z of each layer, list of np.narray (ni, m)
#   where n0 is the number of features
#         ni is the number of neurons for the layer i
#         nj is the number of neurons for the layer i - 1
#         nl is the number of neurons for the last/output layer
#         m is the number of samples in the batch
def forward_propagation(Ws, Bs, X):
  # YOUR CODE HERE #
  layers = len(Ws)
  Zs = [None] * layers
  A = X
  for i in range(layers):
    Zs[i] = Ws[i] @ A + Bs[i]
    A = sigmoid(Zs[i])
  return Zs


# The performance metric of the neural network.
# Y_hat: the prediction, np.narray (nl, m)
# Y: the ground truth, np.narray (nl, m)
# returns the loss for the current batch, float
#   where nl is the number of neurons for the last layer
#         m is the number of samples in the batch
def calculate_loss(Y_hat, Y):
  # YOUR CODE HERE #
  m = Y.shape[1]
  loss = np.sum(-(1 - Y) * np.log(1 - Y_hat) - Y * np.log(Y_hat)) / m
  return loss


# The gradient used to update the weights and biases of the neural network.
# Ws: a list of weight matrices for each layer in the neural network, list of np.narray (ni, nj)
# Bs: a list of bias vectors for each layer in the neural network, list of np.narray (ni, 1)
# X: a batch of features from the data set, np.narray (n0, m)
# Zs: a list of linear combinations Z for each layer in the neural network, list of np.narray (ni, m)
# Y: the ground truth, np.narray (nl, m)
# returns a list of gradients of the loss function with regard to the weights and biases of each layer, tuple (list of np.narray (ni, nj), list of np.narray (ni, 1))
#   where n0 is the number of features
#         ni is the number of neurons for the layer i
#         nj is the number of neurons for the layer i - 1
#         nl is the number of neurons for the last/output layer
#         m is the number of samples in the batch
def backpropagation(Ws, Bs, X, Zs, Y):
  # YOUR CODE HERE #
  layers = len(Ws)

  dloss_dWs = [None] * layers
  dloss_dBs = [None] * layers

  m = X.shape[1]
  
  Y_hat    = sigmoid(Zs[-1])
  dloss_dZ = (Y_hat - Y) / m
  
  for i in reversed(range(layers)):
    Ap = sigmoid(Zs[i-1]) if i > 0 else X
    dZ_dW = Ap.T
    dloss_dW = dloss_dZ @ dZ_dW
    dloss_dB = np.sum(dloss_dZ, axis=1, keepdims=True)

    dloss_dWs[i] = dloss_dW
    dloss_dBs[i] = dloss_dB

    if i != 0:
      dAp_dZp  = Ap * (1 - Ap)
      dZ_dAp   = Ws[i].T
      dloss_dZ = dAp_dZp * (dZ_dAp @ dloss_dZ)

  return dloss_dWs, dloss_dBs


##############################################################################
# The code below should not be changed prior to completing the exercises.    #
# After you pass all the tests you can play with the code below.             #
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
      Zs = forward_propagation(Ws, Bs, X)
      Z[j,i] = Zs[-1]
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
  length = max_x - min_x
  
  padding = 0.1
  min_x_padded = min_x - padding * length
  max_x_padded = max_x + padding * length

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
    Zs    = forward_propagation(Ws, Bs, Xn)
    Y_hat = sigmoid(Zs[-1])

    dloss_dWs, dloss_dBs = backpropagation(Ws, Bs, Xn, Zs, Y)

    for i in range(len(Ws)):
      Ws[i] = Ws[i] - learning_rate * dloss_dWs[i]
      Bs[i] = Bs[i] - learning_rate * dloss_dBs[i]

    # update the graphs every 1000 iterations
    if iteration % 1000 == 0:
      boundary_graph = draw_boundry(Ws, Bs, boundary_X, axs, boundary_graph, boundry_samples)

      loss = calculate_loss(Y_hat, Y)

      draw_loss(iteration, loss, iterations_axis, losses_axis, axs, loss_graph)

      accuracy = np.sum(np.where(Y_hat >= 0.5, 1, 0) == Y) / Y.shape[1]

      print(f"{iteration:{len(str(iterations))}}/{iterations} loss: {loss:0.06f} accuracy: {accuracy:7.2%}")

      fig.canvas.draw()
      fig.canvas.flush_events()
      plt.pause(1)


def run_tests():
  Ws = [
    np.array([[0.18849228, 0.11252503],
              [0.52082511, 0.15898884],
              [0.91090497, 0.25880367]]),

    np.array([[0.03639007, 0.97456206, 0.10809881]])
  ]

  Bs = [
    np.array([[0.29808449],
              [0.13073665],
              [0.77885121]]),

    np.array([[0.74776100]]),
  ]

  X = np.array([[ 0.56803856,  0.45989575, -0.80686698, -0.32368534,  0.62895350],
                [-0.29165671, -0.03532201,  0.55736034, -0.19029145, -0.46055749]])

  Y = np.array([[0, 1, 1, 0, 1]])

  # check the sigmoid activation function
  expected_sigmoid = np.array([[0.49289013, 0.72869778, 0.50879759, 0.71480934, 0.66706825],
                               [0.49285778, 0.72188486, 0.45189955, 0.63578142, 0.69873782]])
  actual_sigmoid = sigmoid(np.array([[-0.0284414 ,  0.98802567,  0.03519401,  0.91885792,  0.69495487],
                                     [-0.02857082,  0.95383044, -0.19299864,  0.55710063,  0.84129466]]))

  assert isinstance(actual_sigmoid, np.ndarray), "sigmoid must return a np.darray"

  assert actual_sigmoid.shape == expected_sigmoid.shape, "sigmoid np.darray shape is incorrect"

  assert np.allclose(actual_sigmoid, expected_sigmoid), "sigmoid np.darray values are incorrect"

  # check the forward_propagation function
  expected_Zs = [
    np.array([[ 0.37233669,  0.38079668,  0.20871328,  0.21565975,  0.36481312],
              [ 0.38021523,  0.3646461 , -0.20088586, -0.06810102,  0.38508792],
              [ 1.22079853,  1.18863107,  0.18811897,  0.4347565 ,  1.23257411]]),
    np.array([[1.4315959 , 1.42738848, 1.2654673 , 1.30422265, 1.43289769]])
  ]

  actual_Zs = forward_propagation(Ws, Bs, X)

  assert isinstance(actual_Zs, list), "forward_propagation must return a list with 2 elements"

  assert len(actual_Zs) == 2, "forward_propagation must return a list with 2 elements"

  assert actual_Zs[0].shape == expected_Zs[0].shape, "forward_propagation 1st layer weights shape is incorrect"

  assert np.allclose(actual_Zs[0], expected_Zs[0]), "forward_propagation 1st layer weights values are incorrect"

  assert actual_Zs[1].shape == expected_Zs[1].shape, "forward_propagation 2nd layer weights shape is incorrect"

  assert np.allclose(actual_Zs[1], expected_Zs[1]), "forward_propagation 2nd layer weights values are incorrect"

  # check the calculate_loss function
  expected_loss = 0.7735457973842816
  actual_loss = calculate_loss(sigmoid(expected_Zs[-1]), Y)

  assert not isinstance(actual_loss, type(None)), "calculate_loss returned None -- make sure to return a value"

  assert np.allclose(actual_loss, expected_loss), "calculate_loss result is incorrect"

  # check the backpropagation function
  expected_gradients = (
    [
      np.array([[ 0.00029814, -0.00073556],
                [ 0.00783999, -0.01977795],
                [ 0.00057957, -0.00196251]]),
      np.array([[0.11265073, 0.10628488, 0.13658134]])],
    [
      np.array([[0.00175871],
                [0.04744071],
                [0.00447059]]), 
      np.array([[0.19750139]])
    ]
  )
  actual_gradients = backpropagation(Ws, Bs, X, expected_Zs, Y)

  assert isinstance(actual_gradients, tuple), "backpropagation must return a tuple with 2 elements"
  
  assert isinstance(actual_gradients[0], list), "backpropagation first tuple element must be a list"

  assert len(actual_gradients[0]) == 2, "backpropagation first tuple element must be a list with 2 elements"

  assert actual_gradients[0][0].shape == expected_gradients[0][0].shape, "backpropagation 1st layer weights gradient shape is incorrect"

  assert np.allclose(actual_gradients[0][0], expected_gradients[0][0]), "backpropagation 1st layer weights gradient values are incorrect"

  assert actual_gradients[0][1].shape == expected_gradients[0][1].shape, "backpropagation 2nd layer weights gradient shape is incorrect"

  assert np.allclose(actual_gradients[0][1], expected_gradients[0][1]), "backpropagation 2nd layer weights gradient values are incorrect"

  assert len(actual_gradients[1]) == 2, "backpropagation second tuple element must be a list with 2 elements"

  assert actual_gradients[1][0].shape == expected_gradients[1][0].shape, "backpropagation 1st layer biases gradient shape is incorrect"

  assert np.allclose(actual_gradients[1][0], expected_gradients[1][0]), "backpropagation 1st layer biases gradient values are incorrect"

  assert actual_gradients[1][1].shape == expected_gradients[1][1].shape, "backpropagation 2nd layer biases gradient shape is incorrect"

  assert np.allclose(actual_gradients[1][1], expected_gradients[1][1]), "backpropagation 2nd layer biases gradient values are incorrect"

  print("All tests passed!")


if __name__ == '__main__':
  # make sure the tests pass before running gradient descent
  run_tests()

  X, Y = read_the_data_set()

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

  # for debugging purposes we don't want random behaviour
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
