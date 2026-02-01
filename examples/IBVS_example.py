from pybotron import *
pi = np.pi



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


cam_d = Camera(pose= H_d)


width=2*cam_d.u0/cam_d.ku;height=2*cam_d.v0/cam_d.kv

r,p,y = np.random.rand(3,)
R_cube = RPY_to_R(r,p,y)
t_cube = t_d + 0.3*(np.random.rand(1,) + 1) * cam_d.principle_axis
H_cube = Rt_to_H(R_cube,t_cube)
vertices = generate_cube(H_cube,0.1)
n = vertices.shape[1]

px_d = cam_d.project_to_pixels(vertices)
px_c = cam_c.project_to_pixels(vertices)


img_d = H_d @  pts_to_homog(cam_d.project_to_image_plane(vertices))
img_d = img_d[:3,:]

s_d = inv(cam_d.K) @ pts_to_homog(px_d)
s_d = s_d.reshape(-1,n)
s_d = s_d[:2,:]

#--------------sim params------------------
sim_time = 10
dt= 0.016
N = int(sim_time/dt)
eps = 1e-1
crit = eps + 1
Kp = 10
#------------------------------------------

#-------------- plot setup --------------
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
cam_img = fig.add_subplot(122)
ax.set_xlim([-1, 1]);   ax.set_ylim([-1, 1]);   ax.set_zlim([-1, 1]);   ax.set_box_aspect([1,1,1])
cam_img.set_xlim(0, cam_c.resolution[0]) ;cam_img.set_ylim(0, cam_c.resolution[1]); cam_img.set_box_aspect(1)
artists = []
cam_scale = 2e-2
#--------------#--------------#--------------#--------------

#----------- static artists-----------------
cam_d.plot_camera_full(pts=vertices, scale = cam_scale, ax=ax, alpha = 0.5 , linestyle='--')
cam_d_artist = cam_d.get_artists()
plot_cube(vertices,ax=ax)

#------------------------------------------

#--------------plot update function---------------
def update(frame):
    global q,artists
    #equal_axes(ax)

    # Update features
    px_c = cam_c.project_to_pixels(vertices)
    s_c = inv(cam_c.K) @ pts_to_homog(px_c)
    s_c = s_c.reshape(-1,n)
    s_c = s_c[:2,:]

    # Compute control law
    e = (s_c-s_d).flatten(order='F')

    L = interaction_matrix(s_c,cam_c.world_to_camera(vertices))

    e_dot =  -Kp * e

    xi = pinv(L) @ e_dot
    xi_out = adj(cam_c.pose) @ xi
    
    # Compute joint update
    J = robot.jacobian()
    q_dot = damped_pinv(J) @ xi_out
    q_dot = clamp(q_dot, 3)
    
    # Update joints
    q = q + dt * q_dot.flatten()
    
    # Update robot body line
    robot.plot(q, ax=ax)

    # Update camera pose
    cam_c.set_pose(robot.get_EE())
    cam_c.plot_camera_full(pts=vertices, scale = cam_scale, ax=ax)

    
    px_c_art = plot_points_2D(px_c,ax=cam_img, anim= True)
    px_d_art = plot_points_2D(px_d,ax=cam_img,color='b', anim= True)

    # Get artists (to update plot with)
    robot_art = robot.get_artists()
    cam_art = cam_c.get_artists()
    
    artists = robot_art + cam_art + cam_d_artist + px_c_art + px_d_art 

    return artists
#------------------------------------------

#----------------------- Animation function
ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True)
plt.show()