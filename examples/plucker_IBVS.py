from pybotron import *
pi = np.pi



robot = UR3e()
q = np.zeros(7)

H_c = robot.get_EE()
cam_c = Camera(pose= H_c)


R_d = RPY_to_R(pi/12, 0,0) @ H_c[:3,:3]
R_d = np.where(abs(R_d) > 1e-5, R_d, 0)
t_d = H_c[:3,3] - 0.3*np.random.rand(3,)

R_d = H_c[:3,:3]
t_d = H_c[:3,3] + np.array([0,-0.1,0]).T
H_d = Rt_to_H(R_d,t_d)

cam_d = Camera(pose= H_d)

l1 = PluckerLine.from_point_direction(cam_d.principle_point+ np.array([0.2,0.1,0]), np.array([0,1,0.5]))
l2 = PluckerLine.from_point_direction(cam_d.principle_point+ np.array([0.2,0.1,0]), np.array([0,1,1]))

lines_list : List[PluckerLine] = [l1,l2]

width=2*cam_d.u0/cam_d.ku;height=2*cam_d.v0/cam_d.kv



#--------------sim params------------------
sim_time = 10
dt= 0.016
N = int(sim_time/dt)
eps = 1e-1
crit = eps + 1
Kp = 100
#------------------------------------------

#-------------- plot setup --------------
fig = plt.figure(figsize=(12, 8))
ax : Axes3D = fig.add_subplot(121, projection='3d', title="3D Scene")
cam_img = fig.add_subplot(122,  title="Camera Image");cam_img.invert_yaxis()
ax.set_xlim([-0.5, 0.5]);   ax.set_ylim([-0.5, 0.5]);   ax.set_zlim([-0, 1]);   ax.set_box_aspect([1,1,1])
cam_img.set_xlim(0, cam_c.resolution[0]) ;cam_img.set_ylim(cam_c.resolution[1], 0); cam_img.set_box_aspect(1)
artists = []
cam_scale = 2e-2
#--------------#--------------#-----------

#----------- static artists-----------------
cam_d.plot_camera(scale = cam_scale, ax=ax, alpha = 0.5 , linestyle='--')
cam_d_artist = cam_d.get_artists()
line_d_artists = []
for l in lines_list:
        p1 = l.point - 0.1 * cam_d.project_line(l).u 
        p2 = p1 + 0.7 * cam_d.project_line(l).u
        P = cam_d.project_to_pixels(p1);Q = cam_d.project_to_pixels(p2)
        line_d_artists += plot_lines_2D(np.hstack([P,Q]),ax=cam_img, size=1,anim = True,linestyle='--',color = 'b')

#------------------------------------------

P = l1.point; Q = l2.point
points = np.vstack([P,Q]).T


px_d = cam_d.project_to_pixels(points)
s_d = inv(cam_d.K) @ pts_to_homog(px_d)
s_d = s_d.reshape(-1,points.shape[1])
s_d = s_d[:2,:]

s_d= np.hstack([s_d.flatten(order='F'), cam_d.chaumette_projection(l1),cam_d.chaumette_projection(l2)])


#--------------plot update function---------------
def update(frame):
    global q,artists
    #equal_axes(ax)
    line_art = []
    for l in lines_list:
        l.plot(ax=ax)
        p1 = l.point - 0.1 * cam_c.project_line(l).u 
        p2 = p1 + 0.7 * cam_c.project_line(l).u
        P1 = cam_c.project_to_pixels(p1);P2 = cam_c.project_to_pixels(p2)
        line_art += plot_lines_2D(np.hstack([P1,P2]),ax=cam_img, size=1,anim = True)

    px_c = cam_c.project_to_pixels(points)
    s_c = inv(cam_c.K) @ pts_to_homog(px_c)
    s_c = s_c.reshape(-1,points.shape[1])
    s_c = s_c[:2,:]

    s_c= np.hstack([s_c.flatten(order='F'),cam_c.chaumette_projection(l1),cam_c.chaumette_projection(l2)])

    # Compute control law
    e = (s_c-s_d)

    L = np.vstack([interaction_matrix(cam_c.world_to_camera(points)),interaction_matrix(l1,form="lines"),interaction_matrix(l2,form="lines")])
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
    cam_c.plot_camera_full(scale = cam_scale, ax=ax)
    cam_c.plot_lines(ax,lines_list, L=0.2, plot_point= False, color= 'r', linestyle="--", linewidth=1)


    # Get artists (to update plot with)
    robot_art = robot.get_artists()
    cam_art = cam_c.get_artists()
    
    artists = robot_art + cam_art + cam_d_artist + line_art + line_d_artists

    
    return artists
#------------------------------------------
#update(0)

#----------------------- Animation function
ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True)
plt.show()