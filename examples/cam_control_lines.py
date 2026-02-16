from pybotron import *
pi = np.pi

H_c = np.eye(4)
cam_c = Camera(pose= H_c)


R_d = RPY_to_R(pi/12, 0,0) @ H_c[:3,:3]
R_d = np.where(abs(R_d) > 1e-5, R_d, 0)
t_d = H_c[:3,3] - 0.2*np.random.rand(3,)

R_d = H_c[:3,:3]
t_d = H_c[:3,3] + np.array([0,-0.1,0]).T
H_d = Rt_to_H(R_d,t_d)

cam_d = Camera(pose= H_d)

T = Rt_to_H(RPY_to_R(0,-np.pi/2,0),np.array([0.01,0,0]))
l1 = PluckerLine.from_point_direction(cam_d.principle_point+ np.array([0.2,0.1,0]), np.array([0,1,0.5]))
l2 = PluckerLine.from_point_direction(cam_d.principle_point+ np.array([0.2,0.1,0]), np.array([0,1,1]))
l1 = l1.transform(T);l2 = l2.transform(T)
lines_list : List[PluckerLine] = [l1,l2]

width=2*cam_d.u0/cam_d.ku;height=2*cam_d.v0/cam_d.kv



#--------------sim params------------------
sim_time = 10
dt= 0.016
N = int(sim_time/dt)
eps = 1e-1
crit = eps + 1
Kp = 5
#------------------------------------------

#-------------- plot setup --------------
fig = plt.figure(figsize=(12, 8))
ax : Axes3D = fig.add_subplot(121, projection='3d', title="3D Scene")
cam_img = fig.add_subplot(122,  title="Camera Image")
ax.set_xlim([-0.3, 0.3]);   ax.set_ylim([-0.3, 0.3]);   ax.set_zlim([-0.3, 0.3]);   ax.set_box_aspect([1,1,1])
cam_img.set_xlim(0, cam_c.resolution[0]) ;cam_img.set_ylim(0, cam_c.resolution[1]); cam_img.set_box_aspect(1)
artists = []
cam_scale = 2e-2
#--------------#--------------#-----------

#----------- static artists-----------------

cam_d.plot_camera(scale = cam_scale, ax=ax, alpha = 0.5 , linestyle='--')
cam_d_artist = cam_d.get_artists()
line_d_artists = []
for l in lines_list:
        p1 = l.point - 0.7 * cam_d.project_line(l).u 
        p2 = p1 + 2 * cam_d.project_line(l).u
        P = cam_d.project_to_pixels(p1);Q = cam_d.project_to_pixels(p2)
        line_d_artists += plot_lines_2D(np.hstack([P,Q]),ax=cam_img, size=1,anim = True,linestyle='--',color = 'b')

#------------------------------------------

s_d = np.hstack([cam_d.chaumette_projection(l1),cam_d.chaumette_projection(l2)])

#--------------plot update function---------------
def update(frame):
    global artists,H_c,H_d
    #equal_axes(ax)
    line_art = []
    for l in lines_list:
        l.plot(ax=ax)
        p1 = l.point - 2 * cam_c.project_line(l).u 
        p2 = p1 + 5 * cam_c.project_line(l).u
        P = cam_c.project_to_pixels(p1);Q = cam_c.project_to_pixels(p2)
        line_art += plot_lines_2D(np.hstack([P,Q]),ax=cam_img, size=1,anim = True)


    s_c= np.hstack([cam_c.chaumette_projection(l1),cam_c.chaumette_projection(l2)])

    # Compute control law
    e = (s_c-s_d).flatten(order='F')

    L = np.vstack([interaction_matrix(l1,form="lines"),interaction_matrix(l2,form="lines")])
    e_dot =  -Kp * e

    xi = pinv(L) @ e_dot
    xi_out = adj(cam_c.pose) @ xi

    delta = twist_to_hat(dt*xi_out)
    
    H_c = expm(delta) @ H_c
    cam_c.set_pose(H_c)
    cam_c.plot_camera_full(scale = cam_scale, ax=ax)
    cam_c.plot_lines(ax,lines_list, L=0.2, plot_point= True, color= 'r', linestyle="--", linewidth=1)

    cam_art = cam_c.get_artists()
    
    artists =  cam_art + cam_d_artist + line_art + line_d_artists
    
    return artists
#------------------------------------------
#update(0)

#----------------------- Animation function
ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True)
plt.show()