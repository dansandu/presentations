import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider


with open('../data/exercise_life.txt', 'r') as f:
  lines =[line.strip().split(' ') for line in f.readlines()]
  X = np.array([float(line[0]) for line in lines]).reshape(-1, 1)
  Y = np.array([float(line[1]) for line in lines]).reshape(-1, 1)


min_x = np.amin(X)
max_x = np.amax(X)

min_y = np.amin(Y)
max_y = np.amax(Y)

padding = 5

fig, axs = plt.subplots()
fig.subplots_adjust(bottom=0.2)

axs.set_title('Model')
axs.set_xlim(min_x - padding, max_x + padding)
axs.set_ylim(min_y - padding, max_y + padding)
axs.set_xlabel('X (hours of exercise per month)')
axs.set_ylabel('Y (life expectency in years)')

axs.scatter(X, Y, marker='x', color='red', linewidth=1)


init_w = (max_y - min_y) / (max_x - min_x)
init_b = init_w * - min_x + min_y

def f(w, b, X):
  return w * X + b

x_space = np.linspace(min_x - padding, max_x + padding, 10)

model, = axs.plot(x_space, f(init_w, init_b, x_space), color='blue')


w_space_center = 0.0
w_space_radius = 4.0

b_space_center = 70.0
b_space_radius = 50.0

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
  fig.canvas.draw_idle()

w_slider.on_changed(on_slider_changed)
b_slider.on_changed(on_slider_changed)

plt.show()
