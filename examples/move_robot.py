from pybotron import *

#---- matplotlib figure setup
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
equal_axes(ax)
#-----------------------------

robot = UR3e()
thetas_0 = np.array([np.pi/8,-np.pi/4,0,0,0,0,np.pi]) #starting joint angles

#---- sim params
sim_time = 2
dt= 0.02
N = int(sim_time/dt)
#----------------

def update(frame):
    global thetas_0
    thetas = thetas_0*(frame)/N
    robot.plot(thetas, ax=ax)


    return robot.get_artists()



ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True, repeat =False)
#fig is the plot
#update is how elements in plot should be updated
#frames is after how many frames the animation stops
#interval is update interval (ie 1/fps)
#blit is for optimization
#repeat is for animation looping
plt.show()