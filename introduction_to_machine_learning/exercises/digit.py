import argparse
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pickle


#########################################################################################################
# Build a neural network model capable of digit recognition using the neural_network_model function.    #
# Change the path_to_dataset function to provide the correct path to your data set.                     #
#                                                                                                       #
# Every model you train will be saved to the models directory with the accuracy shown in the file name. #
# Once you have a model you can make predictions using the --image and --model command line arguments.  #
# The image will be resized for you. You can use your favorite image editing tool to create images.     #
#                                                                                                       #
# To train a model call the script without any arguments:                                               #
#   py digit.py                                                                                         #
#                                                                                                       #
# To make predictions write a prompt with the arguments below:                                          #
#   py digit.py --image=path/to/my/digit.png --model=../models/digit_recognition_98.14.pickle           #
#                                                                                                       #
#########################################################################################################


# Returns the path to the images and labels files from the MNIST data set. Note that the script expects
# these files to be decompressed already.
def path_to_dataset():
  return {
    'train_images': 'path/to/train/images',
    'train_labels': 'path/to/train/labels',
    'test_images': 'path/to/test/images',
    'test_labels': 'path/to/test/labels',
  }


# The neural network model used to recognize digits.
# The data set consists of 28x28 grey scale images therefore any given feature is a 784 dimension vector (n0=784).
# The values of the vectors are between [0,255] and between [-1,1] after normalization.
# returns the weights and biases for each layer, tuple (list of np.narray (ni, nj), list of np.narray (ni, 1))
#   where n0 is the number of features
#         ni is the number of neurons for the layer i
#         nj is the number of neurons for the layer i - 1
def neural_network_model():
  Ws = [
    np.random.rand(200, 784),
    np.random.rand(10, 200),
  ]

  Bs = [
    np.random.rand(200, 1),
    np.random.rand(10, 1),
  ]
  return Ws, Bs


#########################################################################################################
# Besides changing the model you can also change the hyperparameters below to improve your performance. #
#########################################################################################################


# The learning rate should be low enough otherwise the algorithm will not converge but increasing the 
# learning rate can increase the convergence speed.
learning_rate = 0.001


# The weight decay is a regularization technique which tries to reduce the magnitude of the weight 
# parameters and thus helps the model to generalize better.
weight_decay = 0.0001


# The momentum is an increase to the parameters update with regard to the previous update step.
# A part of the previous gradient is transfered to each weight and bias update.
momentum = 0.99


# The number of iterations used to train the model.
# Each epoch passes once through the entire training set.
epochs = 2000


#######################################################################################
# The code below should not be changed prior to completing the exercise.              #
# You can always get a fresh copy of this file so don't worry if you break something. #
#######################################################################################


train_image_count = 60000
test_image_count = 10000
image_width = 28
image_height = 28
batch_size = 100


def sigmoid(Z):
  A = 1.0 / (np.exp(-Z) + 1.0)
  return A


def softmax(Z):
  A = np.exp(Z)
  A = A / np.sum(A, axis=0)
  return A


def forward_propagation(Ws, Bs, X):
  layers = len(Ws)
  As = [None] * layers
  A = X
  for i in range(layers):
    Z = Ws[i] @ A + Bs[i]
    if i + 1 != layers:
      A = sigmoid(Z)
    else:
      A = softmax(Z)
    As[i] = A
  return As


def calculate_loss(Y_hat, Y):
  m = Y.shape[1]
  loss = np.sum(-Y * np.log(Y_hat)) / m
  return loss


def backpropagation(Ws, X, As, Y, weight_decay):
  layers = len(Ws)

  dloss_dWs = [None] * layers
  dloss_dBs = [None] * layers

  m = X.shape[1]
  
  Y_hat    = As[-1]
  dloss_dZ = (Y_hat - Y) / m
  
  for i in reversed(range(layers)):
    Ap = As[i-1] if i > 0 else X

    dZ_dW = Ap.T
    dloss_dW = dloss_dZ @ dZ_dW + weight_decay * Ws[i]
    dloss_dB = np.sum(dloss_dZ, axis=1, keepdims=True)

    dloss_dWs[i] = dloss_dW
    dloss_dBs[i] = dloss_dB

    if i != 0:
      dAp_dZp  = Ap * (1 - Ap)
      dZ_dAp   = Ws[i].T
      dloss_dZ = dAp_dZp * (dZ_dAp @ dloss_dZ)

  return dloss_dWs, dloss_dBs


def read_the_data_set(images_path, labels_path, image_count):
  with open(images_path, 'rb') as f:
    # skip header
    f.read(16)
    raw_data = f.read(image_count * image_width * image_height)
    X = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32).reshape(image_count, image_width * image_height).T

  with open(labels_path, 'rb') as f:
    # skip header
    f.read(8)
    raw_data = f.read(image_count)
    labels = np.frombuffer(raw_data, dtype=np.uint8)
    Y = np.eye(10)[labels].T

  return X, Y


def draw_loss(iterations, losses, axs, loss_graph):
  loss_graph.set_data(iterations, losses)
  axs[1].set_title(f"Loss: {losses[-1]:0.2f}")
  axs[1].relim() 
  axs[1].autoscale_view(True,True,True)


def initialize_graphs(images, labels):
  plt.ion()
  fig, axs = plt.subplots(1, 2)

  image_index = np.random.randint(0, images.shape[1])
  image = images[:,image_index].reshape(image_width, image_height)
  axs[0].set_title(f"Sample #{image_index} Label: {np.argmax(labels[:,image_index])}")
  axs[0].imshow(image)

  loss_graph, = axs[1].plot([], [], color='red')
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel('Loss')

  return fig, axs, loss_graph


def normalize(X):
  return (X - 0.5 * 255) / (0.5 * 255)


def run_gradient_descent(train_X, train_Y, test_X, test_Y, initial_Ws, initial_Bs, batch_size, epochs, learning_rate, weight_decay, momentum):
  train_Xn = normalize(train_X)
  test_Xn  = normalize(test_X)

  train_m = train_X.shape[1]
  test_m  = test_X.shape[1]

  Ws = initial_Ws
  Bs = initial_Bs

  Ws_update = [np.zeros(Ws[i].shape) for i in range(len(Ws))]
  Bs_update = [np.zeros(Bs[i].shape) for i in range(len(Bs))]

  losses = []
  iterations = []
  accuracy = None

  fig, axs, loss_graph = initialize_graphs(train_Xn, train_Y)

  batches = train_m // batch_size + (1 if train_m % batch_size != 0 else 0)
  
  for epoch_index in range(epochs):
    batch_losses = []
    for batch_index in range(batches):
      batch_begin = batch_index * batch_size
      batch_end = min(batch_begin + batch_size, train_m)

      batch_X = train_Xn[:,batch_begin:batch_end]
      batch_Y = train_Y[:,batch_begin:batch_end]

      batch_As = forward_propagation(Ws, Bs, batch_X)

      dloss_dWs, dloss_dBs = backpropagation(Ws, batch_X, batch_As, batch_Y, weight_decay)

      for i in range(len(Ws)):
        Ws_update[i] = momentum * Ws_update[i] + learning_rate * dloss_dWs[i]
        Bs_update[i] = momentum * Bs_update[i] + learning_rate * dloss_dBs[i]

        Ws[i] = Ws[i] - Ws_update[i]
        Bs[i] = Bs[i] - Bs_update[i]

      batch_Y_hat = batch_As[-1]
      batch_loss = calculate_loss(batch_Y_hat, batch_Y)
      batch_losses.append(batch_loss)
  
    iterations.append(epoch_index)
    losses.append(np.mean(batch_losses))
    draw_loss(iterations, losses, axs, loss_graph)

    test_As = forward_propagation(Ws, Bs, test_Xn)
    test_Y_hat = test_As[-1]
    accuracy = np.sum(np.argmax(test_Y_hat, axis=0) == np.argmax(test_Y, axis=0)) / test_m
    print(f"epoch: {epoch_index+1:{len(str(epochs))}}/{epochs} loss: {losses[-1]:.6f} accuracy: {accuracy:7.2%}")

    fig.canvas.draw()
    fig.canvas.flush_events()

  if accuracy != None:
    model_path = f"../models/digit_recognition_{accuracy * 100:.2f}.pickle"
    with open(model_path, 'wb') as f:
      data = {
        "weights": Ws, 
        "biases": Bs,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
      }
      pickle.dump(data, f)


if __name__ == "__main__":
  argument_parser = argparse.ArgumentParser()
  argument_parser.add_argument("--image")
  argument_parser.add_argument("--model",)
  arguments = argument_parser.parse_args()

  image_path = arguments.image
  model_path = arguments.model

  if image_path != None and model_path != None:
    with open(model_path, 'rb') as f:
      data = pickle.load(f)

    Ws = data['weights']
    Bs = data['biases']

    hyperparameters = {k: v for k, v in data.items() if k not in ['weights', 'biases']}
    hyperparameters['weights_shapes'] = [W.shape for W in Ws]
    hyperparameters['biases_shapes'] = [B.shape for B in Ws]

    print("Model was trained with the following hyper-parameters:")
    for k, v in hyperparameters.items():
      print(f"{k}: {v}")

    image = PIL.Image.open(image_path).convert("L").resize((image_width,image_height))
    image = np.array(image.getdata(), dtype=np.uint8).reshape(image_width * image_height, 1)

    Xn = normalize(image)
    As = forward_propagation(Ws, Bs, Xn)
    Y_hat = As[-1]
    prediction = np.argmax(Y_hat)

    fig, axs = plt.subplots()
    axs.set_title(f"Prediction: {prediction}")
    axs.imshow(image.reshape(image_width, image_height))
    plt.show()

  elif image_path == None and model_path == None:
    paths = path_to_dataset()

    train_X, train_Y = read_the_data_set(paths['train_images'], paths['train_labels'], train_image_count)

    test_X, test_Y = read_the_data_set(paths['test_images'], paths['test_labels'], test_image_count)

    Ws, Bs = neural_network_model()

    run_gradient_descent(train_X, train_Y, test_X, test_Y, Ws, Bs, batch_size, epochs, learning_rate, weight_decay, momentum)

    print("Press enter to exit...", end='')
    input()

  else:
    print("You must either supply both the image and the model to make predicitons or neither to train a model")
