from pybotron import *
pi = np.pi

#-------------- plot setup --------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1]);   ax.set_ylim([-1, 1]);   ax.set_zlim([-1, 1]);   ax.set_box_aspect([1,1,1])
artists = []
cam_scale = 2e-2
#--------------#--------------#----------

l = PluckerLine(np.array([0,0,1]),np.array([0,0,0]))

#--------------sim params------------------
sim_time = 10
dt= 0.016
N = int(sim_time/dt)
#------------------------------------------

#--------------plot update function---------
def update(frame):
    global artists

    l.plot(2,ax)

    artists = l.get_artist()

    return artists
#------------------------------------------

#----------------------- Animation function
ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True)
plt.show()