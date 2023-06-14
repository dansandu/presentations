import matplotlib.pyplot as plt
import numpy as np


#################################################################################
# Implement the evaluate_model, calculate_loss and calculate_gradient functions #
# such that all tests pass.                                                     #
#################################################################################


# The model is a simple line and it's used to fit the data.
# w: the slope of the line, float
# b: the bias of the model, float
# X: the feature set representing monthly hours spent exercising, np.narray (1, m)
# returns the predicted life expectancy for all samples in X, np.narray (1, m)
def evaluate_model(w, b, X):
  # YOUR CODE HERE #
  Z = w * X + b
  return Z


# The loss function evaluates how well the model is performing.
# w: the slope of the line, float
# b: the bias of the model, float
# X: the feature set representing monthly hours spent exercising, np.narray (1, m)
# Y: the ground truth representing life expectancy, np.narray (1, m)
# returns the total loss with regard to all samples, float
def calculate_loss(w, b, X, Y):
  # YOUR CODE HERE #
  Y_hat = evaluate_model(w, b, X)
  error = Y_hat - Y
  loss = 0.5 * np.mean(error ** 2)
  return loss


# The gradient of the loss function.
# w: the slope of the line, float
# b: the bias of the model, float
# X: the feature set representing monthly hours spent exercising, np.narray (1, m)
# Y: the ground truth representing life expectancy, np.narray (1, m)
# returns the gradient of the loss function with respect to w and b over all samples, tuple (float, float)
def calculate_gradient(w, b, X, Y):
  # YOUR CODE HERE #
  m = X.shape[1]
  Y_hat = evaluate_model(w, b, X)
  error = Y_hat - Y
  dloss_dw = error @ X.T / m
  dloss_db = np.mean(error)
  return dloss_dw, dloss_db


##############################################################################
# The code below should not be changed prior to completing the exercises.    #
# After you pass all the tests we encourage you to play with the code below. #
##############################################################################


def read_the_data_set():
  with open('../../data/exercise_life.txt', 'r') as f:
    lines =[line.strip().split(' ') for line in f.readlines()]
    X = np.array([float(line[0]) for line in lines]).reshape(1, -1)
    Y = np.array([float(line[1]) for line in lines]).reshape(1, -1)
  return (X, Y)


def initialize_graphs(w, b, X, Y):
  min_x = np.amin(X)
  max_x = np.amax(X)
  min_y = np.amin(Y)
  max_y = np.amax(Y)
  padding = 5

  plt.ion()
  fig, axs = plt.subplots(1, 2)
  boundary_X = np.array([min_x - padding, max_x + padding])
  boundary_graph, = axs[0].plot(boundary_X, evaluate_model(w, b, boundary_X), color='blue')
  axs[0].set_xlim((min_x - padding, max_x + padding))
  axs[0].set_ylim((min_y - padding, max_y + padding))
  axs[0].scatter(X, Y, marker='x', color='red', linewidth=1)
  axs[0].set_title('Model')
  axs[0].set_xlabel('X (hours of exercise per month)')
  axs[0].set_ylabel('Y (life expectency in years)')

  loss_graph, = axs[1].plot([], [], color='red')
  axs[1].set_xlabel('Iterations')
  axs[1].set_ylabel('Loss')
  
  return fig, axs, boundary_graph, loss_graph, boundary_X


def run_gradient_descent(X, Y, initial_w, initial_b, learning_rate, iterations):
  w = initial_w
  b = initial_b

  losses_axis = []
  iterations_axis = []

  fig, axs, boundary_graph, loss_graph, boundary_X = initialize_graphs(w, b, X, Y)

  for i in range(iterations):
    loss = calculate_loss(w, b, X, Y)

    dloss_dw, dloss_db = calculate_gradient(w, b, X, Y)
    w = w - learning_rate * dloss_dw
    b = b - learning_rate * dloss_db

    # update the graphs every 200 iterations
    if i % 200 == 0:
      boundary_graph.set_ydata(evaluate_model(w, b, boundary_X))

      iterations_axis.append(i + 1)
      losses_axis.append(loss)
      loss_graph.set_data(iterations_axis, losses_axis)
      axs[1].set_title(f"Loss: {loss:0.2f}")
      axs[1].relim() 
      axs[1].autoscale_view(True,True,True)

      fig.canvas.draw()
      fig.canvas.flush_events()
      plt.pause(1)


def run_tests():
  w = -3.92
  b = -7.23
  X = np.array([ 4.22472067, 72.11309655, 90.86630719, 94.59906687, 26.70927638,
                73.89344081, 83.48165226, 47.74728446, 62.49155561, 63.41059541]).reshape(1, -1)
  Y = np.array([53.24625157, 66.90703337, 61.34939556, 78.34649885, 75.04485441,
                31.1235417 , 50.68359981, 99.01937705, 45.68523462, 42.61063239]).reshape(1, -1)

  # check the model
  expected_model = np.array([ -23.79090503, -289.91333848, -363.42592418, -378.05834213, -111.93036341, 
                             -296.89228798, -334.47807686, -194.39935508, -252.19689799, -255.79953401]).reshape(1, -1)
  actual_model = evaluate_model(w, b, X)

  assert isinstance(actual_model, np.ndarray), "evaluate_model should return a np.ndarray"

  assert actual_model.shape == expected_model.shape, "evaluate_model returned a np.ndarray with incorrect dimensions"

  assert np.allclose(actual_model, expected_model), "evaluate_model result is incorrect"

  # check the loss
  expected_loss = 53838.77654355495
  actual_loss = calculate_loss(w, b, X, Y)

  assert not isinstance(actual_loss, type(None)), "calculate_loss returned None -- make sure to return a value"

  assert np.allclose(actual_loss, expected_loss), "calculate_loss result is incorrect"

  # check the gradient
  expected_gradient = (-22076.36752275032, -310.4904444732006)
  actual_gradient = calculate_gradient(w, b, X, Y)

  assert isinstance(actual_gradient, tuple), "calculate_gradient should return a tuple"

  assert len(actual_gradient) == 2, "calculate_gradient should return a tuple with to elements"

  assert np.isclose(actual_gradient[0], expected_gradient[0]), "calculate_gradient weight is incorrect"

  assert np.isclose(actual_gradient[1], expected_gradient[1]), "calculate_gradient bias is incorrect"
  
  print("All tests passed!")


if __name__ == '__main__':
  X, Y = read_the_data_set()

  # make sure the tests pass before running gradient descent
  run_tests()

  # you can randomly initialize to try different values
  # w = np.random.rand()
  # b = np.random.rand()

  # for debugging purposes you don't want random behaviour
  w = 0.58 
  b = 0.28

  # The data is not normalized so make sure the learning rate is low enough
  # otherwise the algorithm will not converge.
  # As a bonus exercise try running the algorithm with the data normalized.
  # With the data normalized you can increase the learning rate and the
  # algorithm will converge much faster.
  learning_rate = 0.002

  iterations = 10000

  run_gradient_descent(X, Y, w, b, learning_rate, iterations)

  print("Press enter to exit...", end='')
  input()
