import math
import numpy as np
from numpy import *
        
joint_values = [1.75, 5.76, 2.18, 2.44, 4.54, 0.0]
        
# Joint limits (in radians)
joint_limits = [
    [-2.687, 2.687],
    [-2.618, 2.618],
    [-2.618, 2.618],
    [-2.600, 2.600],
    [-2.530, 2.530],
    [-2.600, 2.600],
]

num_dof = 6

class EndEffector:
    """
    Minimal end-effector pose container.

    This class is used throughout the kinematics code to represent a robot end-effector
    pose (position + orientation).

    Attributes:
        x: x-position (meters).
        y: y-position (meters).
        z: z-position (meters).
        rotx: roll angle about x-axis (radians).
        roty: pitch angle about y-axis (radians).
        rotz: yaw angle about z-axis (radians).
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotx: float = 0.0
    roty: float = 0.0
    rotz: float = 0.0


def rotm_to_euler(R: np.ndarray):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).

    This function assumes the rotation matrix uses the common Z-Y-X convention
    (yaw-pitch-roll composition). The implementation also includes handling for
    near-singular configurations (gimbal lock), where multiple Euler solutions exist.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        A tuple (roll, pitch, yaw) in radians.

    Notes:
        - If `r31` is close to ±1, pitch is near ±90° and the solution is not unique.
        - This function chooses a reasonable representative solution in those cases.
    """
    r11 = R[0,0] if abs(R[0,0]) > 1e-7 else 0.0
    r12 = R[0,1] if abs(R[0,1]) > 1e-7 else 0.0
    r21 = R[1,0] if abs(R[1,0]) > 1e-7 else 0.0
    r22 = R[1,1] if abs(R[1,1]) > 1e-7 else 0.0
    r32 = R[2,1] if abs(R[2,1]) > 1e-7 else 0.0
    r33 = R[2,2] if abs(R[2,2]) > 1e-7 else 0.0
    r31 = R[2,0] if abs(R[2,0]) > 1e-7 else 0.0

    if abs(r31) != 1:
        roll = math.atan2(r32, r33)        
        yaw = math.atan2(r21, r11)
        denom = math.sqrt(r11 ** 2 + r21 ** 2)
        pitch = math.atan2(-r31, denom)
    
    elif r31 == 1:
        # pitch is close to -90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        pitch, yaw = -pi/2, 0.0
        roll = -math.atan2(r12, r22)
    
    elif r31 == -1:
        # pitch is close to 90 deg, i.e. cos(pitch) = 0.0
        # there are an infinitely many solutions, so we choose one possible solution where yaw = 0
        pitch, yaw = pi/2, 0.0
        roll = math.atan2(r12, r22)

    return roll, pitch, yaw


def dh_to_matrix(dh_params: list):
    """
    Convert Denavit–Hartenberg (DH) parameters to a homogeneous transform using the classic
    DH convention.

    Reference: https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters

    Args:
        dh_params: DH parameters [theta, d, a, alpha], where:
            - theta: joint angle (rad)
            - d: link offset along previous z (m)
            - a: link length along current x (m)
            - alpha: link twist about current x (rad)

    Returns:
        4x4 homogeneous transformation matrix.

    Notes:
        This is the "standard" DH transform convention.
    """
    theta, d, a, alpha = dh_params
    return np.array([
        [cos(theta), -sin(theta) * cos(alpha), sin(theta) * sin(alpha), a * cos(theta)],
        [sin(theta), cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a * sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1]
    ])


def euler_to_rotm(rpy: tuple):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.

    This uses Z-Y-X composition (yaw then pitch then roll):
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Args:
        rpy: (roll, pitch, yaw) in radians.

    Returns:
        3x3 rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(rpy[0]), -math.sin(rpy[0])],
                    [0, math.sin(rpy[0]), math.cos(rpy[0])]])
    R_y = np.array([[math.cos(rpy[1]), 0, math.sin(rpy[1])],
                    [0, 1, 0],
                    [-math.sin(rpy[1]), 0, math.cos(rpy[1])]])
    R_z = np.array([[math.cos(rpy[2]), -math.sin(rpy[2]), 0],
                    [math.sin(rpy[2]), math.cos(rpy[2]), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x

def check_joint_limits(theta, theta_limits):
    """
    Checks if the joint angles are within the specified limits.

    Args:
        theta (List[float]): Current joint angles.
        theta_limits (List[List[float]]): Joint limits for each joint.

    Returns:
        bool: True if all joint angles are within limits, False otherwise.
    """
    for i, th in enumerate(theta):
        if not (theta_limits[i][0] <= th <= theta_limits[i][1]):
            return False
    return True

def sample_valid_joints(n_tries: int = 1000):
    """
    Sample a random joint configuration that satisfies the robot's joint limits.

    This is useful for generating random test cases or sanity checks.

    Args:
        robot: Robot model instance that provides:
            - num_dof (int): number of joints
            - joint_limits (list[list[float]]): joint limits in radians/meters
        n_tries: Maximum number of random samples to attempt before failing.

    Returns:
        list[float]: A joint configuration `q` (length = robot.num_dof) that
        satisfies `ut.check_joint_limits(q, robot.joint_limits)`.

    Raises:
        RuntimeError: If no valid configuration is found after `n_tries` attempts.
    """
    for _ in range(n_tries):
        q = [random.uniform(-math.pi, math.pi) for _ in range(num_dof)]
        if check_joint_limits(q, joint_limits):
            return q
    raise RuntimeError("Could not sample valid joint values; check joint limits/ranges.")

def compute_transforms(joint_values):
    """
    Helper to calculate cumulative transformation matrices (H_cumulative)
    and individual joint transforms (Hlist) based on the Kinova robot model.
    
    Args:
        joint_values (list/array): Processed joint angles in radians.
    """
    theta = joint_values
    
    # DH parameters
    # Note: The Kinova model uses 7 frames for 6 joints (Frame 0 is base offset)
    DH = np.array([
        [theta[0]-(pi/2), 0.1283+0.115, 0, pi/2],
        [theta[1]+(pi/2), 0.03, 0.280, pi],
        [theta[2]+(pi/2), 0.02, 0, pi/2],
        [theta[3]+(pi/2), 0.140+0.105, 0, pi/2],
        [theta[4]+pi, 0.0285*2, 0, pi/2],
        [theta[5]+(pi/2), 0.105+0.130, 0, 0]
    ])

    Hlist = [dh_to_matrix(dh) for dh in DH] # Compute transformation matrices for each joint

    # Compute cumulative transformations
    H_cumulative = [np.eye(4)]
    for H in Hlist:
        H_cumulative.append(H_cumulative[-1] @ H)
        
    return H_cumulative, Hlist

def calc_forward_kinematics(joint_values: list, radians=True):
    """
    Calculate Forward Kinematics (FK) based on the given joint angles.

    Args:
        joint_values (list): Joint angles (in radians if radians=True, otherwise in degrees).
        radians (bool): Whether the input angles are in radians (default is False).
    """
    curr_joint_values = joint_values.copy()

    if not radians: # Convert degrees to radians if the input is in degrees
        curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

    # Ensure that the joint angles respect the joint limits
    for i, theta in enumerate(curr_joint_values):
        curr_joint_values[i] = np.clip(theta, joint_limits[i][0], joint_limits[i][1])
    
    H_cumulative, Hlist = compute_transforms(curr_joint_values)

    # Calculate EE position and rotation
    H_ee = H_cumulative[-1]  # Final transformation matrix for EE

    # Set the end effector (EE) position
    ee = EndEffector()
    ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
    
    # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
    rpy = rotm_to_euler(H_ee[:3, :3])
    ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

    return ee, Hlist

def calc_numerical_ik(ee, joint_values, pos_tol=0.005, ori_tol=0.005, ilimit=300):
    """
    Numerical IK with angles wrapped to [-pi, pi] and joint limit enforcement.

    Args:
        ee (EndEffector): Desired end-effector pose.
        joint_values (list[float]): Initial guess for joint angles (radians).
        tol (float, optional): Convergence tolerance. Defaults to 0.002.
        ilimit (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        list[float]: Estimated joint angles in radians, wrapped to [-pi, pi] and within joint limits.
    """
    # unpack end effector position and rotation
    x_target, y_target, z_target, xrot_target, yrot_target, zrot_target = ee.x, ee.y, ee.z, ee.rotx, ee.roty, ee.rotz
    new_joint_values = np.array(joint_values, dtype=float)

    for _ in range(200):  # allow 100 attempted starting configurations
        for _ in range(ilimit): # 100 iterations for each attempted configuration
            # get the end effector position based on the current joint guess and find the error from the desired position
            current_ee, _ = calc_forward_kinematics(new_joint_values)
            pos_error = np.array([x_target, y_target, z_target]) - np.array([current_ee.x, current_ee.y, current_ee.z])
            def angle_diff(a, b):
                d = a - b
                return (d + np.pi) % (2*np.pi) - np.pi

            orient_error = np.array([
                angle_diff(xrot_target, current_ee.rotx),
                angle_diff(yrot_target, current_ee.roty),
                angle_diff(zrot_target, current_ee.rotz)
            ])
            error = np.concatenate((pos_error,orient_error),axis=0)
            # if the error is within tolerance, return the joint angle solution
            if np.linalg.norm(pos_error) <= pos_tol and np.linalg.norm(orient_error) <= ori_tol:
                return new_joint_values
            # get next iteration by updating with inverse jacobian
            #new_joint_values += inverse_jacobian(new_joint_values) @ pos_error
            J = jacobian(new_joint_values)
            J_pos = J # 6x6

            J_pinv = np.linalg.pinv(J_pos)  # 6x6

            new_joint_values = new_joint_values + 0.1*(J_pinv @ error)
            # print(f'Error: {error}')
            # print(f"Joint Values: {new_joint_values}")
            # enforce joint limits
            for i, (low, high) in enumerate(joint_limits):
                new_joint_values[i] = np.clip(new_joint_values[i], low, high)
        # if not converged, return a random configuration and try again
        new_joint_values = np.array(sample_valid_joints(), dtype=float)
    
    print("Fail")
    # return null if not converged
    return np.zeros(len(joint_values))

def calc_inverse_kinematics(target_ee, q_guess=None, tol=1e-4, ilimit=150):
    q = np.array(q_guess if q_guess is not None else [0.0]*6, dtype=float)
    lambda_sq = 0.01  # Damping factor for singularity robustness
    
    # Target pose extraction
    p_targ = np.array([target_ee.x, target_ee.y, target_ee.z])
    R_targ = euler_to_rotm((target_ee.rotx, target_ee.roty, target_ee.rotz))
    for _ in range(100):
        for _ in range(ilimit):
            H_c, _ = compute_transforms(q)
            H_ee = H_c[-1]
            
            # Position Error
            dp = p_targ - H_ee[:3, 3]
            
            # Orientation Error (Skew symmetric matrix)
            R_curr = H_ee[:3, :3]
            R_err = R_targ @ R_curr.T
            do = 0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])

            # Combine position and orientation erros
            error = np.hstack([dp, do])
            if np.linalg.norm(error) < tol:
                return [wraptopi(val) for val in q]

            # Damped Least Squares Update
            J = jacobian(q)
            JJT = J @ J.T + lambda_sq * np.eye(6)
            dq = J.T @ np.linalg.solve(JJT, error)
            q += dq
            limits = np.array(joint_limits)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        q = np.array(sample_valid_joints(), dtype=float)

    return q

def jacobian(joint_values: list):
    """
    Returns the Jacobian matrix for the robot. 

    Args:
        joint_values (list): The joint angles for the robot.

    Returns:
        np.ndarray: The Jacobian matrix (6x6).
    """
    curr_joint_values = joint_values.copy()

    # Ensure that the joint angles respect the joint limits
    for i, theta in enumerate(curr_joint_values):
        curr_joint_values[i] = np.clip(theta, joint_limits[i][0], joint_limits[i][1])
    
    H_cumulative, _ = compute_transforms(curr_joint_values)

    p_ee = H_cumulative[-1][:3, 3] 
    J = np.zeros((6, num_dof)) 
    
    for i in range(num_dof):
        
        transform = H_cumulative[i]
        z_axis = transform[:3, 2] # Z-axis of the frame about which theta[i] rotates?
        
        transform_axis = H_cumulative[i] # Frame defining the Z axis
        
        z_axis = transform_axis[:3, 2]
        p_joint = transform_axis[:3, 3]
        
        J[:3, i] = np.cross(z_axis, (p_ee - p_joint))
        J[3:, i] = z_axis
        
    return J

def inverse_jacobian(joint_values: list):
    """
    Returns the inverse of the Jacobian matrix.

    Returns:
        np.ndarray: The inverse Jacobian matrix.
    """
    return np.linalg.pinv(jacobian(joint_values))

def wraptopi(angle_rad):
    """
    Wrap an angle to the range [-pi, pi).

    Args:
        angle_rad: Angle in radians.

    Returns:
        Equivalent angle in radians in the interval [-pi, pi).
    """
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi