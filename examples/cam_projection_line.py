from pybotron import *
pi = np.pi

H = Rt_to_H(RPY_to_R(-pi/2, 0,0),np.zeros(3))
cam = Camera(pose= H)


l1 = PluckerLine(np.array([1,1,1]),np.array([0.2,0,-0.2]))
l2 = PluckerLine(np.array([1,0,0]),np.array([0,0,-0.2]))



lines_list : List[PluckerLine] = [l1,l2]


#-------------- plot setup --------------
fig = plt.figure(figsize=(12, 8))
ax : Axes3D = fig.add_subplot(121, projection='3d', title="3D Scene")
cam_img = fig.add_subplot(122,  title="Camera Image");cam_img.invert_yaxis()
ax.set_xlim([-0.3, 0.3]);   ax.set_ylim([-0.3, 0.3]);   ax.set_zlim([-0.3, 0.3]);   ax.set_box_aspect([1,1,1])
cam_img.set_xlim(0, cam.resolution[0]) ;cam_img.set_ylim(cam.resolution[1],0); cam_img.set_box_aspect(1)
artists = []
cam_scale = 2e-2
#--------------#--------------#-----------



cam.plot_lines(ax,lines_list, L=0.2, plot_point= True, color= 'r', linestyle="--", linewidth=1)


for l in lines_list:
    l.plot(ax=ax)

    p = l.point - 0.1 * cam.project_line(l).u; q = p + 0.5 * cam.project_line(l).u
    P = cam.project_to_pixels(p);Q = cam.project_to_pixels(q)

    plot_lines_2D(np.hstack([P,Q]),ax=cam_img, size=1)
    plot_lines_2D(np.hstack([P,Q]),ax=cam_img, size=1)


plot_points_3D(p.reshape(-1,1),ax= ax, color='g')
plot_points_3D(q.reshape(-1,1),ax= ax, color='g')

pts = np.vstack([p,q]).T
cam.plot_camera_full(pts,scale = cam_scale, ax=ax)


plt.show()