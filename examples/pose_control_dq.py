from pybotron import *
pi = np.pi

#-------------- plot setup --------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1]);   ax.set_ylim([-1, 1]);   ax.set_zlim([-1, 1]);   ax.set_box_aspect([1,1,1])
artists = []
cam_scale = 2e-2
#--------------#--------------#--------------#--------------

robot = UR3e()
q = np.zeros((1,7)).flatten()

H_c = robot.get_EE()

cam_c = Camera(pose= H_c)


R_d = RPY_to_R(pi/12, 0,0) @ H_c[:3,:3]
R_d = np.where(abs(R_d) > 1e-5, R_d, 0)
t_d = H_c[:3,3] - 0.2*np.random.rand(3,)

R_d = H_c[:3,:3]
t_d = H_c[:3,3] + np.array([0,-0.1,0]).T
H_d = Rt_to_H(R_d,t_d)

dq_d : DualQuaternion = matrix_to_dq(H_d)

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
    equal_axes(ax)

    # Compute error
    H_c = robot.get_EE()
    dq_c : DualQuaternion = matrix_to_dq(H_c)
    dq_err : DualQuaternion = ~dq_d * dq_c
    u, theta = dq_err.rotation_axis_angle()
    R_err, t_err = H_to_Rt(dq_err.to_homogeneous())
    twist_err = np.block([theta*u.T, R_err.T @ t_err])
    
    xi_err = - Kp *  adj(H_d) @ twist_err
    
    # Compute joint update
    J = robot.jacobian()
    q_dot = Kp * damped_pinv(J,damp = 0.001) @ xi_err
    q_dot = clamp(q_dot, 3)
    
    # Update joints
    q = q + dt * q_dot.flatten()
    
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