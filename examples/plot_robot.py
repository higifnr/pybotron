from pybotron import *

#---- matplotlib figure setup
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.5, 0.5]);   ax.set_ylim([-0.5, 0.5]);   ax.set_zlim([-0, 1]);   ax.set_box_aspect([1,1,1])
equal_axes(ax)
#-----------------------------

robot = UR3e()
robot.plot(ax=ax)

plt.show()