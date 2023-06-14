import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from matplotlib import ticker


with open('../data/exercise_life.txt', 'r') as f:
  lines =[line.strip().split(' ') for line in f.readlines()]
  X = np.array([float(line[0]) for line in lines]).reshape(-1, 1)
  Y = np.array([float(line[1]) for line in lines]).reshape(-1, 1)

min_x = np.amin(X)
max_x = np.amax(X)

min_y = np.amin(Y)
max_y = np.amax(Y)

padding = 5

fig, axs = plt.subplots(1, 2)
fig.subplots_adjust(bottom=0.2)

axs[0].set_title('Model')
axs[0].set_xlim(min_x - padding, max_x + padding)
axs[0].set_ylim(min_y - padding, max_y + padding)
axs[0].set_xlabel('X (hours of exercise per month)')
axs[0].set_ylabel('Y (life expectency in years)')

axs[0].scatter(X, Y, marker='x', color='red', linewidth=1)

init_w = (max_y - min_y) / (max_x - min_x)
init_b = init_w * - min_x + min_y

def f(w, b, X):
  return w * X + b


x_space = np.linspace(min_x - padding, max_x + padding, 10)

model, = axs[0].plot(x_space, f(init_w, init_b, x_space), color='blue')

def loss(w, b, X, Y):
  return np.mean((f(w, b, X) - Y) ** 2) / 2.0

w_space_center = 0.0
w_space_radius = 4.0

b_space_center = 70.0
b_space_radius = 50.0

sample_rate = 60

w_space = np.linspace(w_space_center - w_space_radius, w_space_center + w_space_radius, sample_rate)
b_space = np.linspace(b_space_center - b_space_radius, b_space_center + b_space_radius, sample_rate)
loss_space = np.array([loss(w, b, X, Y) for b in b_space for w in w_space]).reshape((sample_rate, sample_rate))

contour = axs[1].contourf(w_space, b_space, loss_space, locator=ticker.LogLocator())
axs[1].set_title('Loss function')
axs[1].set_xlabel('w')
axs[1].set_ylabel('b')
fig.colorbar(contour)

cursor, = axs[1].plot(init_w, init_b, marker='x', ms=10, color='white', linewidth=1)

ax_w = fig.add_axes([0.12, 0.06, 0.7, 0.04])
w_slider = Slider(
  ax=ax_w,
  label='w',
  valmin=w_space_center - w_space_radius,
  valmax=w_space_center + w_space_radius,
  valinit=init_w,
)

ax_b = fig.add_axes([0.12, 0.01, 0.7, 0.04])
b_slider = Slider(
  ax=ax_b,
  label='b',
  valmin=b_space_center - b_space_radius,
  valmax=b_space_center + b_space_radius,
  valinit=init_b,
)

def on_slider_changed(val): 
  model.set_ydata(f(w_slider.val, b_slider.val, x_space))
  cursor.set_xdata([w_slider.val])
  cursor.set_ydata([b_slider.val])
  fig.canvas.draw_idle()

w_slider.on_changed(on_slider_changed)
b_slider.on_changed(on_slider_changed)

plt.show()
