import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
arcdots, = ax.plot([], [], lw=0.1)
x = []
y = []
step = 0
# initialization function: plot the background of each frame
def init():
    arcdots.set_data([], [])
    return arcdots,

# animation function.  This is called sequentially
def animate(i):
    for j in range(50):
        global step
        step += random.uniform(0.01, 0.1)
        angle = step + (math.pi*2)
        a = random.uniform(1.3, 1.7)
        b = random.uniform(0.7, 1.3)
#         a = 1.5 + random.uniform(-0.2, 0.2)
#         b = 1 + random.uniform(-0.2, 0.2)

        h = random.uniform(-0.2, 0.2)
        k = random.uniform(-0.2, 0.2)
        
        x = (h + (a * math.cos(angle))) + random.uniform(-0.2, 0.2)
        y = (k + (b * math.sin(angle))) + random.uniform(-0.2, 0.2)
#         print("x {}, y {}".format(x, y))
        s = random.uniform(0.7, 10.0)
        alpha = random.uniform(0.1, 0.8)
        plt.scatter(x,y, color='gray', s= 2, alpha= alpha)
        if step >= 6.28:
            step = 0
    return arcdots,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
plt.show()

#ellipse rotation
#center = (h, k)
#majax = a
#minax = b
#(((x - h) * cos(angle) + (y - k) * sin(angle)) ** 2) / (a ** 2) + (((x-h) * sin(angle) + (y - k) * cos(angle)) ** 2) / (b ** 2)
#x = center + (a * cos(t) * (cos(rotation), sin(rotation)) + (b * sin(t) * (-sin(rotation), cos(rotation))