# ==============================
# Welcome to pybotron! 
# a Python package meant for making Robotics / Vision simulation easier and matlab-like
# includes: Dual Quaternions, Quaternions, Linear Algebra, Plotting, Plücker Lines
# ==============================

# --- System / Math ---
import numpy as np
from numpy import sin, cos, sqrt
from numpy.linalg import norm, pinv, inv
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation as Rot

# --- Plotting / Visualization ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import cv2


# --- Math functions ---

def pts_to_homog(point : np.ndarray):
    pt = point.copy()
    if pt.ndim == 1:
        pt = pt.reshape(-1,1)
    n = pt.shape[1]
    return np.vstack((pt, np.ones((1, n))))

def skew(vector):
    """Return the skew-symmetric matrix of a 3-vector"""
    v = np.asarray(vector).copy().reshape(3)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def unskew(S):
    w = np.array([S[2,1],S[0,2],S[1,0]]).reshape(3,1)
    return w


def rot_exp(omega, theta):
    w = np.asarray(omega).copy().reshape(3,)
    n = np.linalg.norm(w)
    if n == 0:
        return np.eye(3)
    w = w / n
    K = skew(w)
    return expm(theta*K)

def rodrigues(omega, theta):
    w = np.asarray(omega).copy().reshape(3,)
    n = np.linalg.norm(w)
    if n == 0:
        return np.eye(3)
    w = w / n
    K = skew(w)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)



def twist_to_hat(xi):
    """Return 4x4 matrix form of a twist vector xi = [w,v]^T ."""
    w = np.array(xi[0:3]).copy()
    v = np.array(xi[3:6]).copy()
    return np.block([
        [skew(w), v.reshape(3,1)],
        [np.zeros((1,4))]
    ])

def adj(H : np.ndarray):
    R = H[:3,:3]
    t = H[:3,3]
    H_bar = np.block([
        [R, np.zeros((3,3))],
        [skew(t)@R, R]
    ])

    return H_bar 

def hat_to_twist(xi_hat : np.ndarray):
    R = xi_hat[:3,:3]
    t = xi_hat[:3,3]
    w = unskew(R)
    try:
        xi = np.concat((w.flatten(),t.flatten())).reshape(6,)
    except AttributeError:
        xi = np.concatenate((w.flatten(), t.flatten())).reshape(6,)

    return xi


def expm_twist(unit_twist, theta):
    xi = np.asarray(unit_twist).copy().reshape(6,)
    w = xi[:3]
    v = xi[3:]

    w_norm = np.linalg.norm(w)

    if w_norm == 0:
        # pure translation
        R = np.eye(3)
        t = v * theta
    else:

        R = rodrigues(w,theta)
        t = (np.eye(3) - R) @ np.cross(w,v)

    return np.block([
        [R, t.reshape(3,1)],
        [np.zeros((1,3)), 1]
    ])


def logm_error(H_c,H_d):
    err_hat = logm(inv(H_c) @ H_d)
    xi_err = hat_to_twist(err_hat)
    return xi_err

def logm_se3(H):
    """SE(3) matrix logarithm with branch cut handling"""
    from scipy.linalg import logm as scipy_logm
    
    R = H[:3, :3]
    
    # Check if we're near the branch cut (trace ≈ -1, meaning θ ≈ π)
    if np.trace(R) < -0.9:
        # Near singularity, flip to "other side"
        H_flipped = H.copy()
        H_flipped[:3, 0] *= -1  # Flip first column
        return scipy_logm(H_flipped)
    
    return scipy_logm(H)

def RPY_to_R(roll, pitch, yaw):
    """
    Converts roll, pitch, yaw angles to a 3x3 rotation matrix.
    Angles are in radians.

    Roll  = rotation about X
    Pitch = rotation about Y
    Yaw   = rotation about Z

    ZYX order (yaw-pitch-roll)
    """

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    # ZYX order: yaw → pitch → roll
    R = Rz @ Ry @ Rx
    return R

def H_to_Rt(H: np.ndarray):
    R, t = H[:3,:3], H[:3,3]
    return R, t

def generate_cube(pose = None, a=0.3):
    """
    Returns cube vertices in world coordinates.

    pose : (4, 4) homogeneous transform
    a    : edge length

    Output: (3, 8) array, each column is a vertex
    """

    if pose is None:
        pose = np.eye(4)

    half = a / 2.0

    # Local-frame cube vertices (8 x 3)
    local_vertices = np.array([
        [-half, -half, -half],
        [ half, -half, -half],
        [ half,  half, -half],
        [-half,  half, -half],
        [-half, -half,  half],
        [ half, -half,  half],
        [ half,  half,  half],
        [-half,  half,  half],
    ])

    # Homogeneous coordinates (8 x 4)
    local_vertices_h = np.hstack([
        local_vertices,
        np.ones((8, 1))
    ])

    # Apply pose (4x4 @ 4x8 → 4x8)
    world_vertices_h = pose @ local_vertices_h.T

    # Return 3D coordinates as columns (3 x 8)
    vertices : np.ndarray = world_vertices_h[:3, :]
    return vertices


def clamp(v: np.ndarray, limit: float):
    v_norm = norm(v)
    if v_norm > limit:  
        return v * limit / v_norm
    return v



def interaction_matrix(features : np.ndarray, points : np.ndarray, form ='points', config ='eye_in_hand'):
    """ 
    Inputs
    ---
    ``features`` : camera image features (vector)
    ``points`` : 3D points expressed in cam coord frame (column stacked)
    ``form`` : type of feature that interaction matrix acts on (points, lines..)
    ``config`` : eye-in-hand or eye-to-hand

    Outputs
    ---
    ``L`` : interaction matrix acting on twists of the form [w;v]
    """
    x = features[0,:]
    y = features[1,:]
    Z = points[2,:]

    n = points.shape[1]
    L = np.zeros((2*n, 6))

    if form == "points":
        L[0::2, :] = np.column_stack([
            x*y,    -1-x*x,    y,   -1/Z,   np.zeros(n),    x/Z            
        ])

        L[1::2, :] = np.column_stack([
            1+y*y,  -x*y,   -x, np.zeros(n),    -1/Z,   y/Z            
        ])
    
    if config != "eye_in_hand":
        L[:,:3] *= -1

    return L


# --- Quaternion Math ---

def matrix_to_dq(H: np.ndarray, return_type="DualQuaternion"):
    """Convert 4x4 homogeneous transformation matrix to unit dual quaternion"""
    if H.shape != (4, 4):
        raise ValueError("Expected 4x4 matrix")
    
    R = H[:3, :3]
    t = H[:3, 3]
    
    # Convert rotation matrix to quaternion
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    
    q_r = np.array([w, x, y, z])
    
    # Dual quaternion from translation
    q_t = np.array([0, t[0], t[1], t[2]])
    q_d = 0.5 * quat_mul(q_t, q_r)

    if return_type == "DualQuaternion":
        qR = Quaternion(q_r[0],q_r[1],q_r[2],q_r[3])
        qD = Quaternion(q_d[0],q_d[1],q_d[2],q_d[3])
        return DualQuaternion(qR, qD)
    if return_type == "numpy":
        return np.concatenate([q_r, q_d])
    
    qR = Quaternion(q_r[0],q_r[1],q_r[2],q_r[3])
    qD = Quaternion(q_d[0],q_d[1],q_d[2],q_d[3])
    return DualQuaternion(qR, qD)
    

def dq_to_matrix(dq: np.ndarray):
    """Convert unit dual quaternion (8-vector) to 4x4 matrix"""
    q_r = dq[:4]
    q_d = dq[4:]
    
    # Quaternion to rotation matrix
    w, x, y, z = q_r
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    # Translation from dual part
    q_r_conj = np.array([w, -x, -y, -z])
    q_t = 2 * quat_mul(q_d, q_r_conj)
    t = q_t[1:]
    
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t
    return H

def dq_mul(a: np.ndarray, b: np.ndarray):
    """Multiply two unit dual quaternions (8-vectors)"""
    a_r, a_d = a[:4], a[4:]
    b_r, b_d = b[:4], b[4:]
    
    c_r = quat_mul(a_r, b_r)
    c_d = quat_mul(a_r, b_d) + quat_mul(a_d, b_r)
    
    return np.concatenate([c_r, c_d])

def dq_conj(dq: np.ndarray):
    """Conjugate of dual quaternion (8-vector)"""
    q_r, q_d = dq[:4], dq[4:]
    q_r_conj = np.array([q_r[0], -q_r[1], -q_r[2], -q_r[3]])
    q_d_conj = np.array([q_d[0], -q_d[1], -q_d[2], -q_d[3]])
    return np.concatenate([q_r_conj, q_d_conj])

def dq_to_screw(dq: np.ndarray):
    """Extract screw parameters from unit dual quaternion (8-vector)"""
    H = dq_to_matrix(dq)
    R = H[:3, :3]
    
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1, 1))
    
    if theta > 1e-6:
        u = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) / (2 * np.sin(theta))
    else:
        u = np.array([0, 0, 1])
        theta = 0
    
    return theta, u

def quat_mul(q1: np.ndarray, q2: np.ndarray):
    """Multiply two quaternions [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


# --- Plotting functions ---

def plot_pose(ax: Axes3D, H, length=0.2, lines=None, **kwargs):
    o = H[:3, 3]
    try:
        x = np.concat((o, o + length * H[:3, 0]))
        y = np.concat((o, o + length * H[:3, 1]))
        z = np.concat((o, o + length * H[:3, 2]))
    except AttributeError:
        x = np.concatenate((o, o + length * H[:3, 0]))
        y = np.concatenate((o, o + length * H[:3, 1]))
        z = np.concatenate((o, o + length * H[:3, 2]))

    if lines is None:
        # init
        lx, = ax.plot(
            [x[0], x[3]], [x[1], x[4]], [x[2], x[5]],
            color='r', **kwargs
        )
        ly, = ax.plot(
            [y[0], y[3]], [y[1], y[4]], [y[2], y[5]],
            color='g', **kwargs
        )
        lz, = ax.plot(
            [z[0], z[3]], [z[1], z[4]], [z[2], z[5]],
            color='b', **kwargs
        )
        return [lx, ly, lz]
    else:
        # update
        lines[0].set_data_3d([x[0], x[3]], [x[1], x[4]], [x[2], x[5]])
        lines[1].set_data_3d([y[0], y[3]], [y[1], y[4]], [y[2], y[5]])
        lines[2].set_data_3d([z[0], z[3]], [z[1], z[4]], [z[2], z[5]])
        return lines



def plot_rectangle_normal(center, normal, width, height, ax: Axes3D, lines=None):
    p = np.asarray(center, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)

    # normalize normal
    n /= np.linalg.norm(n)

    # build orthonormal basis
    if abs(np.dot(n, [1.0, 0.0, 0.0])) < 0.9:
        u = np.cross(n, [1.0, 0.0, 0.0])
    else:
        u = np.cross(n, [0.0, 1.0, 0.0])

    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    hw = width / 2.0
    hh = height / 2.0

    corners_local = np.array([
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
        [-hw, -hh]
    ])

    corners_world = (
        p[:, None]
        + u[:, None] * corners_local[:, 0]
        + v[:, None] * corners_local[:, 1]
    )

    x, y, z = corners_world

    if lines is None:
        # init
        line, = ax.plot(x, y, z, "k-", linewidth=0.5)
        return [line]
    else:
        # update
        line = lines[0]
        line.set_data(x, y)
        line.set_3d_properties(z)
        return lines


def plot_cube(vertices, ax : Axes3D, color='b', linewidth=1):
    """
    Plot a wireframe cube.

    vertices : (3, 8) array, each column is a vertex
    color    : matplotlib color (default 'b')
    linewidth: line width (default 1)
    ax       : matplotlib 3D Axes, optional
    """

    vertices = np.asarray(vertices)
    assert vertices.shape == (3, 8), "vertices must be (3, 8)"

    # Row-wise for indexing (8 x 3)
    V = vertices.T

    # Edge list (0-based)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ])

    for i, j in edges:
        v1 = V[i]
        v2 = V[j]
        ax.plot(
            [v1[0], v2[0]],
            [v1[1], v2[1]],
            [v1[2], v2[2]],
            color=color,
            linewidth=linewidth
        )

    # MATLAB-like defaults
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])  # axis equal
    ax.grid(True)


def equal_axes(ax : Axes3D):
    # Get limits
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    # Find ranges
    X = xlim[1] - xlim[0]
    Y = ylim[1] - ylim[0]
    Z = zlim[1] - zlim[0]
    max_range = max(X, Y, Z) / 2

    # Find midpoints
    mx = np.mean(xlim)
    my = np.mean(ylim)
    mz = np.mean(zlim)

    # Set new limits
    ax.set_xlim3d(mx - max_range, mx + max_range)
    ax.set_ylim3d(my - max_range, my + max_range)
    ax.set_zlim3d(mz - max_range, mz + max_range)


def plot_points_3D(points, ax: Axes3D, artist = None, color='r', size=30,**kwargs):
    """
    Plot 3D points as big dots.

    points : (3, N) array, each column is a point
    ax     : optional matplotlib 3D Axes
    color  : point color
    size   : marker size
    """
    points = np.asarray(points)
    assert points.shape[0] == 3, "points must have shape (3, N)"

    if artist is None:
        artist = ax.scatter(points[0, :], points[1, :], points[2, :],
                c=color, s=size, depthshade=True,**kwargs)

        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        return artist
    else:
        artist._offsets3d = (points[0, :], points[1, :], points[2, :])
        return artist


def plot_points_2D(points, ax:Axes3D, color='r', size=30,**kwargs):
    """
    Plot 3D points as big dots.

    points : (2, N) array, each column is a point
    ax     : optional matplotlib 2D Axes
    color  : point color
    size   : marker size
    """
    points = np.asarray(points)
    assert points.shape[0] == 2, "points must have shape (2, N)"

    ax.scatter(points[0, :], points[1, :],
               c=color, s=size,**kwargs)

    ax.set_box_aspect(1)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True)


def render_image(ax :Axes3D, img : np.ndarray, H : np.ndarray , width, height):
    H_img, W_img = img.shape[:2]

    u = np.linspace(-width/2, width/2, W_img)
    v = np.linspace(-height/2, height/2, H_img)
    U, V = np.meshgrid(u, v)
    Z = np.zeros_like(U)

    P = np.vstack([
        U.flatten(),
        V.flatten(),
        Z.flatten(),
        np.ones(U.size)
    ])

    Pw = (H @ P)[:3, :]
    X = Pw[0].reshape(H_img, W_img)
    Y = Pw[1].reshape(H_img, W_img)
    Z = Pw[2].reshape(H_img, W_img)

    ax.plot_surface(X, Y, Z, facecolors=img, shade=False)


def condition_cv_img(img : np.ndarray):
    
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA).astype(float) / 255.0
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float) / 255.0


# --- Class definitions ---

class SimpleRobot:
    def __init__(self, joint_axes, joint_positions, joint_values, TCP = np.eye(4)):
        """
        init args are home config
        ----------------------------
        joint_axes: list of 3D vectors (ωi)
        joint_positions: list of 3D points (qi) through which each axis passes
        M: home configuration (4x4 numpy array)
        """
        self.joint_axes = [np.array(a, dtype=float) for a in joint_axes]
        self.joint_axes = np.array(self.joint_axes).T
        self.joint_positions = [np.array(q, dtype=float) for q in joint_positions]
        self.joint_positions = np.array(self.joint_positions).T
        self.joint_values = np.array(joint_values).flatten()
        self.EE_pose = np.array(TCP, dtype=float)
        self.EE_pose[:3,3] = np.array([self.joint_positions[:,-1]]).flatten()
        self.EE_init = self.EE_pose.copy()
        self.n = len(joint_axes)
        self.artists = {
                            "robot_body": None,
                            "joint_axes": None,
                            "base_frame": None,
                            "EE_frame": None,
                        }


        self.init_axes = np.zeros((6,self.n))
        for i in range(self.n):
            u = self.joint_axes[:,i]
            q = self.joint_positions[:,i]
            m = np.cross(q, u)
            l = np.concatenate((u,m))
            self.init_axes[:,i] = l.reshape(6,)

        self.J = self.init_axes.copy()

    # --------------------------
    def jacobian(self,):
        return self.J
    
    def get_joint_pos(self,):
        return self.joint_positions
    
    def get_joint_axes(self,):
        return self.joint_axes

    def get_EE(self,):
        return self.EE_pose
    
    def get_EE_init(self,):
        return self.EE_init

    def get_artists(self,):
        # i is list item v is dictionary value (list of artists), checking if list is initialized using not None
        return [i for v in self.artists.values() if v is not None for i in v]


    # --------------------------
    def fpk(self, thetas):
        """Compute Forward Position Kinematics using POE."""
        T = np.eye(4)
        points = np.block([[self.joint_positions],
                            [np.ones(self.joint_positions.shape[1])]])
        axes = self.init_axes.copy()
        for i in range(self.n): #order bit weird to avoid ifs
            H_bar = adj(T)
            axes[:,i] = H_bar @ axes[:,i]
            T = expm_twist(axes[:,i], thetas[i]) @ T
            points[:,i+1]= T @ points[:,i+1]

        return points,axes,T

    # --------------------------
    def plot(self, thetas=None, ax=None, show_frames=True,show_axes=False):
        """Plot the robot in 3D space."""
        if ax is None:
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111, projection='3d')

        if thetas is None:
            thetas = self.joint_values

        points,axes,T = self.fpk(thetas)
                          
        self.J = axes
        points = np.array(points)
        self.EE_pose = T @ self.EE_init


        # ----------------- Robot body -----------------
        if self.artists["robot_body"] is None:
            # First time: create the Line3D and store it
            
            artist, = ax.plot(
                points[0,:],
                points[1,:],
                points[2,:],
                'o-',
                linewidth=2,
                markersize=6,
                color='b'
            )

            self.artists["robot_body"] = [artist,]
        else:
            # Already exists: update the data
            line = self.artists["robot_body"][0]
            line.set_data(points[0,:], points[1,:])
            line.set_3d_properties(points[2,:])
        # --------------------------------------------
        #ax.plot(points[0,:], points[1,:], points[2,:], 'o-', linewidth=2, markersize=6, color='b')
        #ax.auto_scale_xyz(points[0,:], points[1,:], points[2,:])


        if show_axes:
            for xi in axes.T:
                u = xi[:3]
                m = xi[3:]
                q = np.cross(u, m)
                start_point = q - u/2  # Start point of the segment
                end_point = q + u/2   # End point of the segment
                points = np.column_stack((start_point, end_point))
                ax.plot(points[0, :], points[1, :], points[2, :], '--', linewidth=0.5, color='r')



        if show_frames:
            #returning function output to the disctionary because at init artists are None which can't be modified by reference
            self.artists["base_frame"] = plot_pose(ax,np.eye(4),lines=self.artists["base_frame"])
            self.artists["EE_frame"] = plot_pose(ax,self.EE_pose,lines=self.artists["EE_frame"])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        #ax.set_box_aspect([1,1,1])
        return ax
    
# --- Custom camera class ---

class Camera:
    def __init__(
        self,
        f=0.1,
        ku=None,
        kv=None,
        resolution=(1280, 720),  # (width, height)
        u0=None,
        v0=None,
        k1=0.0,
        pose=None
    ):
        self.f = float(f)
        self.resolution = resolution
        self.width, self.height = resolution

        # Pixel density defaults (pixels per unit length)
        self.ku = ku if ku is not None else self.width*1e1
        self.kv = kv if kv is not None else self.height*1e1

        # Principal point defaults
        self.u0 = u0 if u0 is not None else self.width / 2.0
        self.v0 = v0 if v0 is not None else self.height / 2.0

        # Distortion (simple radial)
        self.k1 = k1

        # Pose: camera wrt world
        self.pose = np.eye(4) if pose is None else np.asarray(pose, dtype=float)
        self.artists = {"camera_body": None,
                        "image_plane": None,
                        "image_points": None}

    # ------------------------
    # Intrinsics
    # ------------------------
    @property
    def fx(self):
        return self.f * self.ku

    @property
    def fy(self):
        return self.f * self.kv

    @property
    def K(self):
        return np.array([
            [self.fx, self.k1, self.u0],
            [0.0, self.fy, self.v0],
            [0.0, 0.0, 1.0]
        ])

    @property
    def fov_x(self):
        return float(2 * np.atan2(self.width / 2 , self.fx))

    @property
    def fov_y(self):
        return float(2 * np.atan2(self.height /2 , self.fy))
    
    @property
    def principle_axis(self):
        return self.pose[:3,2]
    
    @property
    def principle_point(self):
        return self.pose[:3,3] + self.f * self.principle_axis

    # ------------------------
    # Coordinate manipulation
    # ------------------------
    def world_to_camera(self, pts_abs):
        """
        Pw: 3xN
        returns: 3xN
        """
        N = pts_abs.shape[1]
        pts_h = np.vstack((pts_abs, np.ones((1, N))))
        projected_points = inv(self.pose) @ pts_h
        return projected_points[:3, :]

    def set_pose(self, pose : np.ndarray):
        self.pose = pose

    def get_artists(self,):
        # i is list item v is dictionary value (list of artists), checking if list is initialized using not None
        return [i for v in self.artists.values() if v is not None for i in v]
      

    # ------------------------
    # Projection pipeline
    # ------------------------

    
    def project_to_homog_plane(self, points_abs : np.ndarray):
        try:
            """
            Pw: 3xN 
            returns: homog_pts 3xN
            """
            pts_abs = points_abs.copy().reshape(3,-1)
            pts_cam = self.world_to_camera(pts_abs)  # 3xN
            pts_normalized = pts_cam / pts_cam[2,:]
            return pts_normalized
        except Exception as e:
            print(f"Error projecting to homogenous plane : {e}")

    def project_to_image_plane(self, pts_abs):
        """
        Pw: 3xN 
        returns: img_pts 3xN
        """
        pts_normalized = self.project_to_homog_plane(pts_abs)  # 3xN
        pts_img = self.f * pts_normalized
        return pts_img


    def project_to_pixels(self, pts_abs):
        """
        Image plane (metric) → pixel plane
        """
        pts_img = self.project_to_homog_plane(pts_abs)
        pts_px = self.K @ pts_img
        return pts_px[:2,:]
    
    def plot_pose(self, ax=None, **kwargs):
        return plot_pose(ax,self.pose,**kwargs)


    def plot_img_plane(self, ax=None, **kwargs):
        self.artists["image_plane"] = plot_rectangle_normal(self.principle_point, 
                                                            self.principle_axis, 
                                                            2*self.u0/self.ku,
                                                            2*self.v0/self.kv,
                                                            ax=ax,
                                                            lines=self.artists["image_plane"])
        
    def plot_points_3D(self, points, ax: Axes3D, color='r', size=30,**kwargs):
        #return type of plot_points_3D is one object and not a list, requiring this if statement
        if self.artists["image_points"] is None:
            self.artists["image_points"]=[plot_points_3D(points, ax=ax, color=color, size=size,**kwargs),]
        else:
            self.artists["image_points"]=[plot_points_3D(points, ax=ax,artist=self.artists["image_points"][0], color=color, size=size,**kwargs),]
        #requires more fixing (points don't update in real time)


    
    def plot_camera(self, scale=0.05, ax=None, **kwargs):
        """
        Plot a wireframe camera and return its artists.
        """

        pose = np.asarray(self.pose)
        assert pose.shape == (4, 4), "pose must be 4x4"

        plot_kw = dict(color='k', linewidth=1)
        plot_kw.update(kwargs)

        if ax is None:
            ax = plt.gca()
            if not isinstance(ax, Axes3D):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

        artists = []
        s = scale

        # ---------- Frustum ----------
        cam_points = np.array([
            [0,  0,  0],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s],
        ]).T

        cam_h = np.vstack([cam_points, np.ones((1, cam_points.shape[1]))])
        world = (pose @ cam_h)[:3, :]
        C = world[:, 0]
        P = world[:, 1:]

        for i in range(4):
            line, = ax.plot(
                [C[0], P[0, i]],
                [C[1], P[1, i]],
                [C[2], P[2, i]],
                **plot_kw
            )
            artists.append(line)

        for i in range(4):
            j = (i + 1) % 4
            line, = ax.plot(
                [P[0, i], P[0, j]],
                [P[1, i], P[1, j]],
                [P[2, i], P[2, j]],
                **plot_kw
            )
            artists.append(line)

        # ---------- Camera body ----------
        body = np.array([
            [-s, -s, -2.5*s],
            [ s, -s, -2.5*s],
            [ s,  s, -2.5*s],
            [-s,  s, -2.5*s],
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0],
        ]).T

        body_h = np.vstack([body, np.ones((1, 8))])
        V = (pose @ body_h)[:3, :].T

        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7)
        ]

        for i, j in edges:
            line, = ax.plot(
                [V[i,0], V[j,0]],
                [V[i,1], V[j,1]],
                [V[i,2], V[j,2]],
                **plot_kw
            )
            artists.append(line)

        # ---------- Camera frame ----------
        frame_artists = self.plot_pose(ax, length=2*scale, **kwargs)
        if frame_artists:
            artists += frame_artists
        self.artists["camera_body"] = artists
        return artists
    

        
    def plot_camera_full(self, pts :np.ndarray = None, scale=0.05, ax=None, **kwargs ):
        """
        ``pts`` : points in world frame
        """
        self.plot_camera(scale=scale,ax=ax, **kwargs )
        self.plot_img_plane(ax)
        if pts is not None:
            img = self.pose @ np.vstack([self.project_to_image_plane(pts),np.ones((1,pts.shape[1]))])
            img = img[:3,:]
            self.plot_points_3D(img,size=10,ax=ax)

#------------------Quaternion---------------------------------------
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.v = np.array([self.x, self.y, self.z], dtype=float)

    def to_array(self):
        return np.array([self.w,self.x,self.y,self.z])
        
    def __pos__(self):
        return self
    
    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)
    
    # Eucledian norm
    def norm(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    # Conjugate
    def __invert__(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    # Explicit conjugate method
    def star(self):
        return ~self

    # Quaternion addition
    def __add__(self,other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        
        elif isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return self + q

        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in addition") 
        
    def __radd__(self,other):
        if isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return self + q
        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in addition") 
            
    
    # Quaternion subtraction
    def __sub__(self,other):
        if isinstance(other, Quaternion):
            return self + (-other)
        elif isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return self - q
        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in subtraction") 
        
    def __radd__(self,other):
        if isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return q - self
        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in subtraction")

    # Quaternion multiplication or scalar multiplication
    def __mul__(self, other):

        if isinstance(other, Quaternion):
            w1, x1, y1, z1 = self.w, self.x, self.y, self.z
            w2, x2, y2, z2 = other.w, other.x, other.y, other.z
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return Quaternion(w, x, y, z)
        
        elif isinstance(other, (int, float)):
            return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
        
        elif isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return self * q

        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in multiplication") 

    # Reflected multiplication: float * quaternion
    def __rmul__(self, other):

        if isinstance(other, (int, float)):
            return self * other
        
        elif isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return self * q

        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in multiplication") 

    # Quaternion division or scalar division
    def __truediv__(self, other):

        if isinstance(other, Quaternion):
            return self * other.inverse()
        
        elif isinstance(other, (int, float)):
            return Quaternion(self.w/other, self.x/other, self.y/other, self.z/other)
        
        elif isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return self / q

        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in division") 

    # Reflected division: float / quaternion
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return self.inverse() * other
        
        elif isinstance(other, np.ndarray):
            v = other.flatten(order="F")
            q = Quaternion(v[0],v[1],v[2],v[3])
            return q / self

        else:
            raise TypeError("Only Quaterions and numpy vectors are allowed in division") 

    # Quaternion inverse
    def inverse(self):
        n2 = self.norm()**2
        if n2 == 0:
            raise ZeroDivisionError("Cannot invert a zero quaternion")
        conj = ~self
        return Quaternion(conj.w / n2, conj.x / n2, conj.y / n2, conj.z / n2)

    # Rotation matrix (3x3)
    def to_rot_matrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y**2 + z**2),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ], dtype=float)
    
    def copy(self):
        return Quaternion(self.w,self.x,self.y,self.z)

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    

# --- Dual Quaternion ---

class DualQuaternion:
    """
    dq = qR + eps qD
    qR, qD are Quaternion
    Works for non-unit dual quaternions
    """

    def __init__(self, qR, qD):
        if isinstance(qR,np.ndarray):
            self.qR = Quaternion(qR[0],qR[1],qR[2],qR[3])
        if isinstance(qD,np.ndarray):
            self.qD = Quaternion(qD[0],qD[1],qD[2],qD[3])
            
        self.qR = qR
        self.qD = qD

    # ---------- constructors ----------

    @staticmethod
    def from_quat_and_translation(q, t):
        t_q = Quaternion(0.0, *t)
        qd = 0.5 * (t_q * q)
        return DualQuaternion(q, qd)

    def copy(self):
        return DualQuaternion(self.qR, self.qD)
    
    def to_array(self,form="vector"):
        if form == "vector":
            return np.append(self.qR.to_array(), self.qD.to_array())
        else:
            return np.hstack(self.qR.to_array().reshape(-1,1),self.qD.to_array().reshape(-1,1))

    # ---------- updaters ----------

    def from_quat_and_translation(self, q, t):
        t_q = Quaternion(0.0, *t)
        qd = 0.5 * (t_q * q)
        self.qR = q
        self.qD = qd

    # ---------- unary ops ----------

    def __pos__(self):
        return self

    def __neg__(self):
        return DualQuaternion(-(self.qR), -(self.qD))

    # spatial conjugation
    def __invert__(self):
        return DualQuaternion(self.qR.star(), self.qD.star())

    def star(self):
        return ~self

    # dual conjugation
    def bar(self):
        return DualQuaternion(self.qR, -(self.qD))

    # total conjugation
    def total_conj(self):
        return self.bar().star()

    # ---------- addition ----------

    def __add__(self, other):

        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.qR + other.qR,
                                  self.qD + other.qD)
        
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            return DualQuaternion(self.qR + qR,
                                  self.qD + qD)
        
        else:
            raise TypeError("Only DualQuaterions and numpy vectors / matrices are allowed in addition") 
        
    def __radd__(self, other):
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            return DualQuaternion(self.qR + qR, self.qD + qD)
        
        else:
            raise TypeError("Only DualQuaterions and numpy vectors / matrices are allowed in addition") 

    def __sub__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.qR - other.qR,
                                  self.qD - other.qD)
        
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            return DualQuaternion(self.qR - qR,
                                  self.qD - qD)
        
        else:
            raise TypeError("Only DualQuaterions and numpy vectors / matrices are allowed in subtraction") 
    
    def __rsub__(self, other):
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            return DualQuaternion(self.qR - qR, self.qD - qD)
        
        else:
            raise TypeError("Only DualQuaterions and numpy vectors / matrices are allowed in subtraction") 

    # ---------- multiplication ----------

    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            qr = self.qR * other.qR
            qd = self.qR * other.qD + self.qD * other.qR
            return DualQuaternion(qr, qd)
        elif isinstance(other, (int, float)):
            return DualQuaternion(self.qR * other,
                                  self.qD * other)
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            dq = DualQuaternion(qR,qD)
            return self*dq

        else:
            raise TypeError("Only DualQuaterions, floats/ints, and numpy vectors / matrices are allowed in multiplication") 

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            dq = DualQuaternion(qR,qD)
            return dq*self

        else:
            raise TypeError("Only DualQuaterions, floats/ints, and numpy vectors / matrices are allowed in multiplication") 

    # ---------- inverse ----------

    def inverse(self):
        """
        General inverse: valid for non-unit dual quaternions
        """
        qr_inv = self.qR.inverse()
        qd_inv = -(qr_inv * self.qD * qr_inv)
        return DualQuaternion(qr_inv, qd_inv)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self * other.inverse()
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            dq = DualQuaternion(qR,qD)
            return self * dq.inverse()
        
        else:
            raise TypeError("Only DualQuaterions, floats/ints, and numpy vectors / matrices are allowed in division") 


    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return other * self.inverse()
        elif isinstance(other, np.ndarray):
            if other.ndim == 1:
                qR = Quaternion(other[0],other[1],other[2],other[3])
                qD = Quaternion(other[4],other[5],other[6],other[7])
            else:
                v = other.flatten(order="F")
                qR = Quaternion(v[0],v[1],v[2],v[3])
                qD = Quaternion(v[4],v[5],v[6],v[7])

            dq = DualQuaternion(qR,qD)
            return dq * self.inverse()
        
        else:
            raise TypeError("Only DualQuaterions, floats/ints, and numpy vectors / matrices are allowed in division") 

    # ---------- action on point ----------

    def __matmul__(self, p):
        """
        Apply transform to a 3D point
        p: 3-vector
        """
        p_q = Quaternion(0.0, *p)
        p_dq = DualQuaternion(Quaternion(1,0,0,0), p_q)
        res = self * p_dq * self.inverse()
        return res.qD.v

    # ---------- matrices ----------

    def to_homogeneous(self):
        """
        4x4 homogeneous transform
        Handles non-unit qR correctly
        """
        n2 = self.qR.norm()**2

        # rotation from normalized qR
        qn : Quaternion = self.qR / np.sqrt(n2)
        R = qn.to_rot_matrix()

        # translation
        t = 2 * (self.qD * self.qR.star()).v / n2

        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = t
        return T

    # ---------- geometry ----------

    def translation(self):
        n2 = self.qR.norm()**2
        return 2 * (self.qD * self.qR.star()).v / n2

    def rotation_axis_angle(self, threshold = 1e-8):
        qn : Quaternion = self.qR / self.qR.norm()
        angle = 2 * np.arccos(qn.w)
        s = np.linalg.norm(qn.v)

        if s < threshold:
            return np.array([1.0, 0.0, 0.0]), 0.0

        return qn.v / s, angle

    def screw_parameters(self):
        """
        Works for general rigid transforms
        """
        axis, angle = self.rotation_axis_angle()
        t = self.translation()

        if angle == 0.0:
            return axis, np.zeros(3), np.inf

        pitch = np.dot(axis, t) / angle
        moment = 0.5 * np.cross(t, axis)
        return axis, moment, pitch

    def __repr__(self):
        return f"DualQuaternion(qR={self.qR}, qD={self.qD})"



# --- Subclasses ---

# --- Child class for UR3e ---
class UR3e(SimpleRobot):
    def __init__(self):
        x,y,z = np.array([1, 0, 0]),np.array([0, 1, 0]),np.array([0, 0, 1])
        axes = [
            z,
            x,
            z,
            -x,
            x,
            z,
            x,
        ]
        # qi positions from literature/examples (you may refine)
        positions = [
            [0, 0, 0],
            [0, 0, 0.152],
            [0.120, 0, 0.152],
            [0.120, 0, 0.152+0.244],
            [0.120-0.093, 0, 0.152+0.244+0.213],
            [0.120-0.093+0.083, 0, 0.152+0.244+0.213],
            [0.120-0.093+0.083, 0, 0.152+0.244+0.213+0.083],
            [0.120-0.093+0.083+0.082, 0, 0.152+0.244+0.213+0.083]
        ]

        
        EE_frame_wrt_base = np.eye(4)
        R_mat = Rot.from_euler('xyz', [0,np.pi/2,0]).as_matrix()
        EE_frame_wrt_base[:3,:3] = R_mat  # example home TCP pos
        EE_frame_wrt_base[:3,3] = np.array([0.120-0.093+0.083+0.082, 0, 0.152+0.244+0.213+0.083])
        joint_values = np.zeros(7)
        super().__init__(axes, positions, joint_values,EE_frame_wrt_base)
        self.home_joints = np.zeros(6)
