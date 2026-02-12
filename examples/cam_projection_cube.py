from pybotron import *
pi = np.pi

H = Rt_to_H(RPY_to_R(-pi/2, 0,0),np.zeros(3))
cam = Camera(pose= H)

r,p,y = np.random.rand(3,)
R_cube = RPY_to_R(r,p,y)
t_cube = 0.3*(np.random.rand(1,) + 1) * cam.principle_axis
H_cube = Rt_to_H(R_cube,t_cube)
vertices = generate_cube(H_cube,0.1)
n = vertices.shape[1]

px_c = cam.project_to_pixels(vertices)

#-------------- plot setup --------------
fig = plt.figure()
ax : Axes3D = fig.add_subplot(121, projection='3d', title="3D Scene")
cam_img = fig.add_subplot(122,  title="Camera Image")
ax.set_box_aspect([1,1,1])
cam_img.set_xlim(0, cam.resolution[0]) ;cam_img.set_ylim(0, cam.resolution[1]); cam_img.set_box_aspect(1)
artists = []
cam_scale = 2e-2
#--------------#--------------#-----------


cam.plot_camera_full(scale = cam_scale, ax=ax)
v_c = transform_points(cam.project_to_image_plane(vertices), cam.pose)
plot_points_3D(v_c,ax=ax, size=5)
plot_points_2D(px_c,ax=cam_img, anim= True)
plot_cube(vertices,ax=ax)
equal_axes(ax)

plt.show()