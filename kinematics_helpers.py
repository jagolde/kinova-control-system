import math
import numpy as np
from numpy import *

l1, l2, l3, l4, l5, l6, l7 = 0.156, 0.128, 0.410, 0.208, 0.105, 0.105, 0.0615
        
joint_values = [1.75, 5.76, 2.18, 2.44, 4.54, 0.0]
        
# Joint limits (in radians)
joint_limits = [
    [0, 2*pi],
    [0, 2*pi],
    [0, 2*pi],
    [0, 2*pi],
    [0, 2*pi],
    [0, 2*pi],
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

def calc_inverse_kinematics(self, ee, joint_values=None, soln=0):
        """
        Calculates the analytical inverse position kinematics for the 6-DOF Kinova manipulator.
        Utilizes kinematic decoupling via the spherical wrist to compute the closed-form 
        solution governed by the explicitly requested geometric root.
        """
        p_ee = np.array([ee.x, ee.y, ee.z])
        R_06 = euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
        d_6 = self.l6 + self.l7
        
        # Wrist Position
        p_wrist = p_ee - d_6 * (R_06 @ np.array([0, 0, 1]))
        wx, wy, wz = p_wrist[0], p_wrist[1], p_wrist[2]

        solutions = []

        theta1_configs = [
            (-atan2(wy, wx), np.sqrt(wx**2 + wy**2)),
            (-atan2(-wy, -wx), -np.sqrt(wx**2 + wy**2))
        ]

        for theta_1, r in theta1_configs:
            s = wz - (self.l1 + self.l2) # Height relative to shoulder
            
            # Law of Cosines for Elbow (beta) and Shoulder (alpha)
            l_proximal = self.l3
            l_distal = self.l4 + self.l5
            L_sq = r**2 + s**2
            
            # Check reachability
            numerator = l_proximal**2 + l_distal**2 - L_sq
            denominator = 2 * l_proximal * l_distal
            if abs(numerator) > abs(denominator):
                continue

            cos_beta = numerator / denominator
            beta = np.arccos(cos_beta)

            # 2. Elbow Config (Up vs Down)
            for theta_3 in [np.pi - beta, -(np.pi - beta)]:
                
                # alpha: Interior angular offset 
                alpha = np.arctan2(l_distal * np.sin(theta_3), l_proximal + l_distal * np.cos(theta_3))

                # gamma: Absolute angular trajectory
                gamma = np.arctan2(s, r)

                # theta_2
                theta_2 = (np.pi / 2) - (gamma - alpha)
                
                # Calculate Rotation of Frame 3 (R_03)
                q_temp = [theta_1, theta_2, theta_3, 0, 0, 0]
                H_cumulative, _ = self._compute_transforms(q_temp)
                R_03 = H_cumulative[4][:3, :3]
                
                R_36 = R_03.T @ R_06
                
                # Wrist Config (Positive vs Negative sine for theta_5)
                c5 = -R_36[2, 2]
                s5_mag = np.sqrt(np.clip(1 - c5**2, 0, 1))
                
                for s5 in [s5_mag, -s5_mag]:
                    theta_5 = atan2(s5, c5)
                    
                    if abs(s5) > 1e-6:
                        theta_4 = atan2(-R_36[1, 2], -R_36[0, 2])
                        theta_6 = atan2(-R_36[2, 1], -R_36[2, 0])
                    else:
                        # Gimbal lock
                        theta_4 = 0
                        theta_6 = atan2(R_36[1, 0], R_36[0, 0])

                    candidate_q = [theta_1, theta_2, theta_3, theta_4, theta_5, theta_6]
                    candidate_q = [self.normalize_angle(q) for q in candidate_q]

                    # Check limits
                    if check_joint_limits(candidate_q, self.joint_limits):
                        solutions.append(candidate_q)

        # Sort solutions by error
        def get_error(q):
            fk_ee, _ = self.calc_forward_kinematics(q)
            return np.linalg.norm(np.array([ee.x, ee.y, ee.z]) - np.array([fk_ee.x, fk_ee.y, fk_ee.z]))
        
        solutions.sort(key=get_error)

        if not solutions:
            return [0, 0, 0, 0, 0, 0]

        # If a specific solution index is requested, return it
        if soln < len(solutions):
            return solutions[soln]
        
        # Return best solution
        return solutions[0]

def normalize_angle(angle):
    """
    Normalize an angle to the range (-pi, pi].
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

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