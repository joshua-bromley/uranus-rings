import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

time = np.linspace(0,100,100)
X = np.cos(time/10)
Y = np.sin(time/10)


def update(t):
    ax.cla()

    x = X[t]
    y = Y[t]
    z = 5

    ax.scatter(x, y, z, s = 100, marker = 'o')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 10)


fig = plt.figure(dpi=100)
ax = fig.add_subplot(projection='3d')

ani = FuncAnimation(fig = fig, func = update, frames = 100, interval = 100)

plt.show()