import matplotlib.pyplot as plt
import numpy as np


################################################################################
# Implement the evaluate_model_without_activation, sigmoid, calculate_loss and #
# calculate_gradient functions such that all tests pass.                       #
################################################################################


# The evaluation of the model but WITHOUT the sigmoid activation function.
# Primarily used to draw the decision boundary of the model.
# W: the weight parameters, np.narray (1, 2)
# b: the bias of the model, float
# X: the feature set for disk activity, np.narray (2, m)
# returns the position of each sample relative to the decision boundary, np.narray (1, m)
def evaluate_model_without_activation(W, b, X):
  # YOUR CODE HERE #
  return


# The activation function mapping values from R space to (0, 1) giving us the
# probability whether the process is malicious or not.
# Z: the position of each sample relative to the decision boundary, np.narray (1, m)
# returns the probability for each value inside Z, np.narray (1, m)
def sigmoid(Z):
  # YOUR CODE HERE #
  return


# The loss function evaluates how well the model is performing.
# Y_hat: the prediction whether the process is malicious or not, np.narray (1, m)
# Y: the ground truth whether the process is malicious or not, np.narray (1, m)
# returns the total loss with regard to all samples, float
def calculate_loss(Y_hat, Y):
  # YOUR CODE HERE #
  return


# The gradient of the loss function.
# X: the feature set for disk activity, np.narray (2, m)
# Y_hat: the prediction whether the process is malicious or not, np.narray (1, m)
# Y: the ground truth whether the process is malicious or not, np.narray (1, m)
# returns the gradient of the loss function with respect to W and b over all samples, tuple (np.narray (1, 2), float)
def calculate_gradient(X, Y_hat, Y):
  # YOUR CODE HERE
  return


##############################################################################
# The code below should not be changed prior to completing the exercises.    #
# After you pass all the tests you can play with the code below.             #
##############################################################################


def read_the_data_set():
  with open('../data/processes.txt', 'r') as f:
    data = np.array([[float(val) for val in line.strip().split(' ')] for line in f.readlines()]).T
    X = data[0:2,:].reshape(2, -1)
    Y = data[2,:].reshape(1, -1)
  return (X, Y)


def draw_boundry(W, b, boundary_X, axs, boundary_graph, samples):
  Z = np.zeros((samples, samples))
  for i in range(samples):
    for j in range(samples):
      x1, x2 = boundary_X[0,i], boundary_X[0,j]
      X = np.array([x1, x2]).reshape(-1, 1)
      Z[j,i] = evaluate_model_without_activation(W, b, X)
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


def initialize_graphs(W, b, X, Y, boundry_samples):
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

  boundary_graph = draw_boundry(W, b, boundary_X, axs, None, boundry_samples)

  return fig, axs, boundary_graph, loss_graph, boundary_X


def normalize(X):
  Minimum = np.amin(X, axis=1, keepdims=True)
  Maximum = np.amax(X, axis=1, keepdims=True)
  return (X - 0.5 * (Minimum + Maximum)) / (0.5 * (Maximum - Minimum))


def run_gradient_descent(X, Y, initial_W, initial_b, learning_rate, iterations):
  # fixes the exploding gradient problem and speeds up learning
  Xn = normalize(X)

  W = initial_W
  b = initial_b

  losses_axis = []
  iterations_axis = []

  boundry_samples = 50
  fig, axs, boundary_graph, loss_graph, boundary_X = initialize_graphs(W, b, Xn, Y, boundry_samples)

  for iteration in range(iterations):
    Y_hat = sigmoid(evaluate_model_without_activation(W, b, Xn))

    loss = calculate_loss(Y_hat, Y)

    dloss_dW, dloss_db = calculate_gradient(Xn, Y_hat, Y)
    W = W - learning_rate * dloss_dW
    b = b - learning_rate * dloss_db

    # update the graphs every 10 iterations
    if iteration % 10 == 0:
      boundary_graph = draw_boundry(W, b, boundary_X, axs, boundary_graph, boundry_samples)

      draw_loss(iteration, loss, iterations_axis, losses_axis, axs, loss_graph)

      accuracy = np.sum(np.where(Y_hat >= 0.5, 1, 0) == Y) / Y.shape[1]

      print(f"{iteration:{len(str(iterations))}}/{iterations} loss: {loss:0.06f} accuracy: {accuracy:7.2%}")

      fig.canvas.draw()
      fig.canvas.flush_events()
      plt.pause(1)


def run_tests():
  W = np.array([[0.17, -2.5]])
  b = -9.3
  X = np.array([[ 4.22472067, 72.11309655, 90.86630719, 94.59906687, 26.70927638],
                [73.89344081, 83.48165226, 47.74728446, 62.49155561, 63.41059541]])
  Y = np.array([[1, 1, 0, 1, 0]])

  # check the model without sigmoid activation function
  expected_model_without_activation = np.array([[-193.31539951, -205.74490424, -113.22093893, -149.44704766, -163.28591154]])
  actual_model_without_activation = evaluate_model_without_activation(W, b, X)

  assert isinstance(actual_model_without_activation, np.ndarray), "evaluate_model_without_activation must return a np.darray"

  assert actual_model_without_activation.shape == expected_model_without_activation.shape, "evaluate_model_without_activation np.darray shape is not correct"

  assert np.allclose(actual_model_without_activation, expected_model_without_activation), "evaluate_model_without_activation np.darray values are not correct"
  
  # check the sigmoid activation function
  expected_sigmoid = np.array([[7.51478224e-22, 0.276878195, 0.5, 0.824913732, 1.0]])
  actual_sigmoid = sigmoid(np.array([[-48.64, -0.96, 0.0, 1.55, 79.35]]))

  assert isinstance(actual_sigmoid, np.ndarray), "sigmoid must return a np.darray"

  assert actual_sigmoid.shape == expected_sigmoid.shape, "sigmoid np.darray shape is incorrect"

  assert np.allclose(actual_sigmoid, expected_sigmoid), "sigmoid np.darray values are incorrect"

  Y_hat = sigmoid(expected_model_without_activation)

  # check the loss
  expected_loss = 109.70147028094001
  actual_loss = calculate_loss(Y_hat, Y)

  assert not isinstance(actual_loss, type(None)), "calculate_loss is returning None -- make sure to return a real value"

  assert np.allclose(actual_loss, expected_loss), "calculate_loss return value is incorrect"

  # check the gradient
  expected_gradient = (np.array([[-34.18737682, -43.97332974]]), -0.6)
  actual_gradient = calculate_gradient(X, Y_hat, Y)

  assert isinstance(actual_gradient, tuple), "calculate_gradient must return a tuple with 2 elements"

  assert len(actual_gradient) == 2, "calculate_gradient must return a tuple with 2 elements"

  assert actual_gradient[0].shape == expected_gradient[0].shape, "calculate_gradient weights shape is incorrect"

  assert np.allclose(actual_gradient[0], expected_gradient[0]), "calculate_gradient weights values are incorrect"

  assert np.isclose(actual_gradient[1], expected_gradient[1]), "calculate_gradient bias values are incorrect"
  
  print("All tests passed!")


if __name__ == '__main__':
  # make sure the tests pass before running gradient descent
  run_tests()

  X, Y = read_the_data_set()

  # we can randomly initialize to try different values
  # W = np.random.rand(1, X.shape[0])
  # b = np.random.rand()

  # for debugging purposes we don't want random behaviour
  W = np.array([[0.51, 0.04]])
  b = 0.28

  # the learning rate should be low enough otherwise the algorithm will not converge
  # but increasing the learning rate can increase the convergence speed
  learning_rate = 0.2

  iterations = 500

  run_gradient_descent(X, Y, W, b, learning_rate, iterations)

  print("Press enter to exit...", end='')
  input()
