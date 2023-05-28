import matplotlib.pyplot as plt
import numpy as np


def sigmoid(Z):
  return 1.0 / (np.exp(-Z) + 1.0)


###########################################################################
# Implement the evaluate_model_without_activation, calculate_loss and     #
# calculate_gradient functions such that all tests pass.                  #
###########################################################################


# The evaluation of the model but WITHOUT the sigmoid activation function.
# Primarily used to draw the decision boundary of the model.
# W: the features parameters, np.narray (1, 5)
# b: the bias of the model, float
# X: the feature set for disk activity, np.narray (5, m)
# returns the position of each sample relative to the decision boundary, np.narray (1, m)
def evaluate_model_without_activation(W, b, X):
  # YOUR CODE HERE #
  return W @ X + b


# The loss function evaluates how well the model is performing.
# W: the features parameters, np.narray (1, 5)
# b: the bias of the model, float
# X: the feature set for disk activity, np.narray (5, m)
# Y: the ground truth whether the process is malicious or not, np.narray (1, m)
# returns the total loss with regard to all samples, float
def calculate_loss(W, b, X, Y):
  # YOUR CODE HERE #
  A = sigmoid(evaluate_model_without_activation(W, b, X))
  return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))


# The gradient of the loss function.
# W: the features parameters, np.narray (1, 5)
# b: the bias of the model, float
# X: the feature set for disk activity, np.narray (5, m)
# Y: the ground truth whether the process is malicious or not, np.narray (1, m)
# returns the gradient of the loss function with respect to W and b over all samples, tuple (np.narray (1, 5), float)
def calculate_gradient(W, b, X, Y):
  # YOUR CODE HERE
  error = sigmoid(evaluate_model_without_activation(W, b, X)) - Y
  m = X.shape[1]
  return error @ X.T / m, np.mean(error)


##############################################################################
# The code below should not be changed prior to completing the exercises.    #
# After you pass all the tests we encourage you to play with the code below. #
##############################################################################


def read_the_data_set():
  with open('../../data/processes.txt', 'r') as f:
    data = np.array([[float(val) for val in line.strip().split(' ')] for line in f.readlines()]).T
    X = data[0:2,:].reshape(2, -1)
    
    # synthesize new features x1^2, x1 * x2, x2^2
    # the model will be more complex and fit the data better
    X = np.concatenate((X, X[[0]] ** 2, X[[0]] * X[[1]], X[[1]] ** 2), axis=0)

    Y = data[2,:].reshape(1, -1)
  return (X, Y)


def draw_boundry(W, b, X1, X2, axs, boundary_graph, samples):
  Z = np.zeros((samples, samples))
  for i in range(samples):
    for j in range(samples):
      x1, x2 = X1[i], X2[j]
      X  = np.array([x1, x2, x1**2, x1 * x2, x2**2]).reshape(-1, 1)
      Z[i,j] = evaluate_model_without_activation(W, b, X)
  if boundary_graph != None:
    for tp in boundary_graph.collections:
      tp.remove()
  boundary_graph = axs[0].contour(X1, X2, Z, levels=[0], colors='blue')
  return boundary_graph


def draw_loss(iteration, loss, iterations_axis, losses_axis, axs, loss_graph):
  iterations_axis.append(iteration + 1)
  losses_axis.append(loss)
  loss_graph.set_data(iterations_axis, losses_axis)
  axs[1].set_title(f"Loss: {loss:0.2f}")
  axs[1].relim() 
  axs[1].autoscale_view(True,True,True)


def initialize_graphs(W, b, X, Y, boundry_samples):
  min_x = np.amin(X, axis=1, keepdims=True)
  max_x = np.amax(X, axis=1, keepdims=True)
  padding = 0.1

  plt.ion()
  fig, axs = plt.subplots(1, 2)

  axs[0].set_xlim((min_x[0][0] * (1 - padding), max_x[0][0] * (1 + padding)))
  axs[0].set_ylim((min_x[1][0] * (1 - padding), max_x[1][0] * (1 + padding)))
  axs[0].scatter(X[0,Y[0]==1], X[1,Y[0]==1], marker='x', color='red', linewidth=1)
  axs[0].scatter(X[0,Y[0]==0], X[1,Y[0]==0], marker='o', color='green', fc='none', ec='green', linewidth=1, s=60)
  axs[0].set_title('Malicious process detection')
  axs[0].set_xlabel('X1 (normalized average written bytes per second)')
  axs[0].set_ylabel('X2 (normalized average erased bytes per second)')

  loss_graph, = axs[1].plot([], [], color='red')
  axs[1].set_xlabel('Iterations')
  axs[1].set_ylabel('Loss')

  boundary_X1 = np.linspace(min_x[0][0] * (1 - padding), max_x[0][0] * (1 + padding), boundry_samples)
  boundary_X2 = np.linspace(min_x[1][0] * (1 - padding), max_x[0][0] * (1 + padding), boundry_samples)

  boundary_graph = draw_boundry(W, b, boundary_X1, boundary_X2, axs, None, boundry_samples)

  return fig, axs, boundary_graph, loss_graph, boundary_X1, boundary_X2


def normalize(X):
  Minimum = np.amin(X, axis=1, keepdims=True)
  Maximum = np.amax(X, axis=1, keepdims=True)
  return (X - 0.5 * (Minimum + Maximum)) / (0.5 * (Maximum - Minimum))


def run_gradient_descent(X, Y, initial_W, initial_b, learning_rate, iterations):

  # fixes exploding gradient problems and speeds up learning
  Xn = normalize(X)

  W = initial_W
  b = initial_b

  losses_axis = []
  iterations_axis = []

  boundry_samples = 50
  fig, axs, boundary_graph, loss_graph, boundary_X1, boundary_X2 = initialize_graphs(W, b, Xn, Y, boundry_samples)

  for iteration in range(iterations):
    loss = calculate_loss(W, b, Xn, Y)

    derror_dw, derror_db = calculate_gradient(W, b, Xn, Y)
    W = W - learning_rate * derror_dw
    b = b - learning_rate * derror_db

    if iteration % 1000 == 0:
      boundary_graph = draw_boundry(W, b, boundary_X1, boundary_X2, axs, boundary_graph, boundry_samples)

      draw_loss(iteration, loss, iterations_axis, losses_axis, axs, loss_graph)

      fig.canvas.draw()
      fig.canvas.flush_events()
      plt.pause(1)


def run_tests():
  W = np.array([[0.17, -2.5, 1.2, -0.32, -2.11]])
  b = -9.3
  X = np.array([[1.19962312, 6.88143946, 8.14048233, 5.84065383, 8.26203831, 5.83940764, 6.15791693, 5.12961967],
                [3.21297537, 0.58654561, 1.78214275, 1.00458577, 1.35757949, 6.89915558, 1.18594044, 5.49019764],
                [6.10376722, 9.95553085, 9.19912784, 5.26573257, 6.20965673, 5.66017636, 9.10013595, 1.89252893],
                [6.36220075, 2.91357901, 0.36290271, 4.6785111 , 5.82226162, 7.72630861, 1.7594032 , 2.87858249],
                [4.47765813, 9.24002417, 4.67822537, 6.72846728, 9.89125599, 3.84542251, 6.24841454, 1.4517947 ]])
  Y = np.array([[1, 1, 0, 1, 0, 0, 0, 1]])

  # check the model without sigmoid activation function
  expected_model_without_activation = np.array([[-21.28774472, -18.07867858, -11.31970587, -20.1938637 , 
                                                 -26.57148799, -29.34923827, -14.04500579, -23.86685725]])
  actual_model_without_activation = evaluate_model_without_activation(W, b, X)

  assert isinstance(actual_model_without_activation, np.ndarray), "evaluate_model_without_activation must return a np.darray"

  assert actual_model_without_activation.shape == expected_model_without_activation.shape, "evaluate_model_without_activation np.darray shape is not correct"

  assert np.allclose(actual_model_without_activation, expected_model_without_activation), "evaluate_model_without_activation np.darray values are not correct"
  
  # check the loss
  expected_loss = 10.428394650367867
  actual_loss = calculate_loss(W, b, X, Y)

  assert not isinstance(actual_loss, type(None)), "calculate_loss is returning None -- make sure to return a real value"

  assert np.allclose(actual_loss, expected_loss), "calculate_loss return value is incorrect"

  # check the gradient
  expected_gradient = (np.array([[-2.38140404, -1.28678523, -2.90218007, -2.10410844, -2.7372353 ]]), -0.49999838216637127)
  actual_gradient = calculate_gradient(W, b, X, Y)

  assert isinstance(actual_gradient, tuple), "calculate_gradient must return a tuple with 2 elements"

  assert len(actual_gradient) == 2, "calculate_gradient must return a tuple with 2 elements"

  assert actual_gradient[0].shape == expected_gradient[0].shape, "calculate_gradient weights shape is incorrect"

  assert np.allclose(actual_gradient[0], expected_gradient[0]), "calculate_gradient weights values is incorrect"

  assert np.isclose(actual_gradient[1], expected_gradient[1]), "calculate_gradient bias values is incorrect"
  
  print("All tests passed!")


if __name__ == '__main__':
  X, Y = read_the_data_set()

  # make sure the tests pass before running gradient descent
  run_tests()

  # you can randomly initialize to try different values
  # W = np.random.rand(1, X.shape[0])
  # b = np.random.rand()

  # for debugging purposes you don't want random behaviour
  W = np.linspace(-1.0, 1.0, X.shape[0]).reshape(1, -1)
  b = 0.28

  # the learning rate should be low enough otherwise the algorithm will not converge
  # but increasing the learning rate can increase the convergence speed
  learning_rate = 0.2

  iterations = 40000

  run_gradient_descent(X, Y, W, b, learning_rate, iterations)

  print("Press enter to exit...", end='')
  input()
