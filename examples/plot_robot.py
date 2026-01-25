from pybotron import *

#---- matplotlib figure setup
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
equal_axes(ax)
#-----------------------------

robot = UR3e()
robot.plot(ax=ax)

plt.show()