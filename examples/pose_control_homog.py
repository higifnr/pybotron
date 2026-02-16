from pybotron import *
pi = np.pi

#-------------- plot setup --------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-0.3, 0.3]);   ax.set_ylim([-0.3, 0.3]);   ax.set_zlim([0, 0.6]);   ax.set_box_aspect([1,1,1])
artists = []
cam_scale = 2e-2
#--------------#--------------#--------------#--------------

robot = UR3e()
q = np.zeros(7)

H_c = robot.get_EE()
cam_c = Camera(pose= H_c)


R_d = RPY_to_R(pi/12, 0,0) @ H_c[:3,:3]
R_d = np.where(abs(R_d) > 1e-5, R_d, 0)
t_d = H_c[:3,3] - 0.2*np.random.rand(3,)

R_d = H_c[:3,:3]
t_d = H_c[:3,3] + np.array([0,-0.1,0]).T
H_d = np.block([[R_d,t_d.reshape(3,1)],
                [0,0,0,1]])


cam_d = Camera(pose= H_d)


#--------------sim params------------------
sim_time = 10
dt= 0.016
N = int(sim_time/dt)
eps = 1e-1
crit = eps + 1
Kp = 100
#------------------------------------------


#----------- static artists-----------------
cam_d.plot_camera(scale = cam_scale, ax=ax, alpha = 0.5 , linestyle='--')
static_cam = cam_d.get_artists()
#------------------------------------------

#--------------plot update function---------------
def update(frame):
    global q, artists, static_cam, Kp

    # Compute error
    H_c = robot.get_EE()
    err_hat = logm(np.linalg.inv(H_c) @ H_d)
    xi_err = hat_to_twist(err_hat)

    xi_out = adj(H_c)@xi_err
    
    # Compute joint update
    J = robot.jacobian()
    q_dot = Kp * damped_pinv(J) @ xi_out
    q_dot = clamp(q_dot, 3)
    
    # Update joints
    q[:] = q + dt * q_dot.flatten()
    
    # Update robot body line
    robot.plot(q, ax=ax)

    cam_c.set_pose(robot.get_EE())
    cam_c.plot_camera(scale = cam_scale, ax=ax)


    robot_art = robot.get_artists()
    cam_art = cam_c.get_artists()
    
    artists = robot_art + cam_art + static_cam

    return artists
#------------------------------------------

#----------------------- Animation function
ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True)
plt.show()