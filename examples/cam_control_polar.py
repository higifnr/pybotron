from pybotron import *
pi = np.pi


H_c = Rt_to_H(RPY_to_R(-pi/2, 0,0),np.zeros(3))
cam_c = Camera(pose= H_c)

r,p,y = np.random.rand(3,)
R_d = RPY_to_R(r,p,y) @ H_c[:3,:3]
R_d = np.where(abs(R_d) > 1e-5, R_d, 0)
t_d = H_c[:3,3] - 0.2*np.random.rand(3,)
H_d = Rt_to_H(R_d,t_d)


cam_d = Camera(pose= H_d)


width=2*cam_d.u0/cam_d.ku;height=2*cam_d.v0/cam_d.kv

r,p,y = np.random.rand(3,)
R_cube = RPY_to_R(r,p,y) #RPY_to_R(r,p,y)
t_cube = t_d + 0.3*(np.random.rand(1,) + 1) * cam_d.principle_axis
H_cube = Rt_to_H(R_cube,t_cube)
vertices = generate_cube(H_cube,0.1)
n = vertices.shape[1]

px_d = cam_d.project_to_pixels(vertices)
px_c = cam_c.project_to_pixels(vertices)


img_d = H_d @  pts_to_homog(cam_d.project_to_image_plane(vertices))
img_d = img_d[:3,:]



#--------------sim params------------------
sim_time = 10
dt= 0.016
N = int(sim_time/dt)
eps = 1e-1
crit = eps + 1
Kp = 10
#------------------------------------------

#-------------- plot setup --------------
fig = plt.figure(figsize=(12, 8))
ax : Axes3D = fig.add_subplot(121, projection='3d', title="3D Scene")
cam_img = fig.add_subplot(122,  title="Camera Image");cam_img.invert_yaxis()
ax.set_xlim([-0.3, 0.3]);   ax.set_ylim([-0.3, 0.3]);   ax.set_zlim([-0.3, 0.3]);   ax.set_box_aspect([1,1,1])
cam_img.set_xlim(0, cam_c.resolution[0]) ;cam_img.set_ylim(cam_c.resolution[1],0); cam_img.set_box_aspect(1)
artists = []
cam_scale = 2e-2
#--------------#--------------#--------------#--------------

#----------- static artists-----------------
cam_d.plot_camera_full(pts=vertices, scale = cam_scale, ax=ax, alpha = 0.5 , linestyle='--')
cam_d_artist = cam_d.get_artists()
plot_cube(vertices,ax=ax)

#------------------------------------------

s_d = inv(cam_d.K) @ pts_to_homog(px_d)
s_d = cam_d.world_to_camera(vertices)

X = s_d[0,:]
Y = s_d[1,:]
Z = s_d[2,:]
x = X/Z
y = Y/Z
rho = np.sqrt(x**2 + y**2)
theta = np.atan2(y,x)
s_d = np.vstack([rho, theta])


#--------------plot update function---------------
def update(frame):
    global artists,H_c,H_d
    #equal_axes(ax)


    px_c = cam_c.project_to_pixels(vertices)
    s_c = inv(cam_c.K) @ pts_to_homog(px_c)
    s_c = cam_c.world_to_camera(vertices)

    X = s_c[0,:]
    Y = s_c[1,:]
    Z = s_c[2,:]
    x = X/Z
    y = Y/Z
    rho = np.sqrt(x**2 + y**2)
    theta = np.atan2(y,x)
    s_c = np.vstack([rho, theta])


    # Compute control law
    e = (s_c-s_d).flatten(order='F')
    e[1::2] = wrap_angles(e[1::2]) #to avoid big rotations
    

    L = interaction_matrix(cam_c.world_to_camera(vertices),form = 'polar')

    e_dot =  -Kp * e

    xi = pinv(L) @ e_dot
    xi_out = adj(cam_c.pose) @ xi

    delta = twist_to_hat(dt*xi_out)
    
    H_c = expm(delta) @ H_c
    cam_c.set_pose(H_c)
    cam_c.plot_camera_full(scale = cam_scale, ax=ax)

    px_c_art = plot_points_2D(px_c,ax=cam_img, anim= True)
    px_d_art = plot_points_2D(px_d,ax=cam_img,color='b', anim= True)

    cam_art = cam_c.get_artists()
    
    artists =  cam_art + cam_d_artist + px_c_art + px_d_art 
    
    return artists
#------------------------------------------
#update(0)

#----------------------- Animation function
ani = FuncAnimation(fig, update, frames=N, interval=dt*1000, blit=True)
plt.show()